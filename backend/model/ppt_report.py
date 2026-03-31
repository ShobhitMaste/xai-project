"""
Run a curated testcase pack through the saved model and export PPT-friendly charts.

Usage (from backend/):
    python model/ppt_report.py
    python model/ppt_report.py --cases data/ppt_demo_cases.json --output_dir reports/ppt_demo
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd

try:
    from model.eval_golden import load_decision_threshold
    from model.explain import explain, get_emotion_signals
    from model.guardrails import apply_conservative_postcheck
    from model.model_loader import get_device, load_model
    from model.predict import predict
    from utils.preprocessing import clean_text
except ModuleNotFoundError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from model.eval_golden import load_decision_threshold
    from model.explain import explain, get_emotion_signals
    from model.guardrails import apply_conservative_postcheck
    from model.model_loader import get_device, load_model
    from model.predict import predict
    from utils.preprocessing import clean_text


def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_cases(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _accepted(case: Dict[str, Any], label: str) -> bool:
    accept = case.get("accept")
    if accept:
        return label in accept
    return label == case["expected"]


def _run_case(
    case: Dict[str, Any],
    model,
    tokenizer,
    device,
    decision_threshold: float,
) -> Dict[str, Any]:
    text = case["text"]
    cleaned = clean_text(text)

    prediction = predict(
        cleaned,
        model,
        tokenizer,
        device,
        decision_threshold=decision_threshold,
    )
    explanations = explain(
        cleaned,
        model,
        tokenizer,
        device,
        prediction_context=prediction,
    )
    final_prediction = apply_conservative_postcheck(cleaned, prediction, explanations)

    if final_prediction != prediction:
        explanations = explain(
            cleaned,
            model,
            tokenizer,
            device,
            prediction_context=final_prediction,
        )

    agreement = explanations.get("agreement", {})
    emotion_signals = explanations.get("emotion_signals") or get_emotion_signals(cleaned)
    rationale = explanations.get("rationale", "")
    actual = final_prediction["label"]
    passed = _accepted(case, actual)

    return {
        "id": case["id"],
        "category": case.get("category", "uncategorized"),
        "text": text,
        "expected": case["expected"],
        "accept": ", ".join(case.get("accept", [])),
        "predicted": actual,
        "pass": passed,
        "probability": float(final_prediction.get("probability", 0.0)),
        "is_uncertain": bool(final_prediction.get("is_uncertain", False)),
        "recommended_action": final_prediction.get("recommended_action", ""),
        "agreement_score": float(agreement.get("score", 0.0)),
        "consensus_words": ", ".join(agreement.get("consensus_words", [])),
        "dominant_emotion": emotion_signals.get("dominant_emotion", "none"),
        "rationale": rationale,
    }


def _export_summary(results_df: pd.DataFrame, metadata: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    total = int(len(results_df))
    passed = int(results_df["pass"].sum())
    failed = int(total - passed)
    accuracy = round(passed / max(1, total), 4)

    category_rows: List[Dict[str, Any]] = []
    for category, group in results_df.groupby("category"):
        category_rows.append(
            {
                "category": category,
                "cases": int(len(group)),
                "passed": int(group["pass"].sum()),
                "failed": int(len(group) - group["pass"].sum()),
                "accuracy": round(float(group["pass"].mean()), 4),
                "avg_probability": round(float(group["probability"].mean()), 4),
                "avg_agreement": round(float(group["agreement_score"].mean()), 4),
            }
        )

    summary = {
        "description": metadata.get("description", ""),
        "model_threshold": metadata["decision_threshold"],
        "total_cases": total,
        "passed_cases": passed,
        "failed_cases": failed,
        "accuracy": accuracy,
        "category_summary": category_rows,
        "failed_case_ids": results_df.loc[~results_df["pass"], "id"].tolist(),
    }

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    notes_path = os.path.join(output_dir, "ppt_talking_points.md")
    with open(notes_path, "w", encoding="utf-8") as f:
        f.write("# PPT Talking Points\n\n")
        f.write(f"- Total curated cases evaluated: **{total}**\n")
        f.write(f"- Passed cases: **{passed}**\n")
        f.write(f"- Accuracy on curated demo pack: **{accuracy * 100:.1f}%**\n")
        f.write(f"- Serving threshold used: **{metadata['decision_threshold']:.2f}**\n")
        if failed:
            f.write(f"- Failure IDs to mention honestly: **{', '.join(summary['failed_case_ids'])}**\n")
        else:
            f.write("- No failures on the curated demo pack.\n")

    return summary


def _set_plot_defaults() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")


def _save_overall_chart(summary: Dict[str, Any], output_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), dpi=180)
    labels = ["Passed", "Failed"]
    values = [summary["passed_cases"], summary["failed_cases"]]
    colors = ["#2E8B57", "#D1495B"]
    bars = ax.bar(labels, values, color=colors, width=0.55)
    ax.set_title("Curated Demo Testcases: Pass vs Fail", fontsize=16, fontweight="bold")
    ax.set_ylabel("Number of Cases")
    ax.text(
        0.98,
        0.95,
        f"Accuracy: {summary['accuracy'] * 100:.1f}%",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=13,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#F3F4F6", "edgecolor": "#D1D5DB"},
    )
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.05, str(value), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "01_overall_pass_fail.png"), bbox_inches="tight")
    plt.close(fig)


def _save_category_chart(category_df: pd.DataFrame, output_dir: str) -> None:
    category_df = category_df.sort_values("accuracy", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=180)
    bars = ax.barh(category_df["category"], category_df["accuracy"] * 100, color="#2563EB")
    ax.set_title("Accuracy by Scenario Category", fontsize=16, fontweight="bold")
    ax.set_xlabel("Accuracy (%)")
    ax.set_xlim(0, 100)
    for bar, value in zip(bars, category_df["accuracy"] * 100):
        ax.text(value + 1, bar.get_y() + bar.get_height() / 2, f"{value:.1f}%", va="center")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "02_category_accuracy.png"), bbox_inches="tight")
    plt.close(fig)


def _save_confusion_chart(results_df: pd.DataFrame, output_dir: str) -> None:
    labels = ["Not Stressed", "Uncertain", "Stressed"]
    matrix = pd.crosstab(
        pd.Categorical(results_df["expected"], categories=labels),
        pd.Categorical(results_df["predicted"], categories=labels),
        dropna=False,
    )
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=180)
    im = ax.imshow(matrix.values, cmap="Blues")
    ax.set_title("Expected vs Predicted Labels", fontsize=16, fontweight="bold")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=20, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Expected")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(int(matrix.iloc[i, j])), ha="center", va="center", color="#111827")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "03_confusion_matrix.png"), bbox_inches="tight")
    plt.close(fig)


def _save_probability_chart(results_df: pd.DataFrame, decision_threshold: float, output_dir: str) -> None:
    label_colors = {
        "Stressed": "#D1495B",
        "Not Stressed": "#2E8B57",
        "Uncertain": "#D97706",
    }
    ordered = results_df.sort_values("probability", ascending=True)
    colors = [label_colors.get(label, "#6B7280") for label in ordered["predicted"]]

    fig, ax = plt.subplots(figsize=(10, 7), dpi=180)
    bars = ax.barh(ordered["id"], ordered["probability"] * 100, color=colors)
    ax.axvline(decision_threshold * 100, color="#111827", linestyle="--", linewidth=1.5, label="Decision threshold")
    ax.set_title("Stress Probability by Testcase", fontsize=16, fontweight="bold")
    ax.set_xlabel("Stress Probability (%)")
    ax.set_ylabel("Case ID")
    ax.set_xlim(0, 100)
    for bar, value in zip(bars, ordered["probability"] * 100):
        ax.text(value + 1, bar.get_y() + bar.get_height() / 2, f"{value:.1f}%", va="center")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "04_probability_by_case.png"), bbox_inches="tight")
    plt.close(fig)


def _save_agreement_chart(results_df: pd.DataFrame, output_dir: str) -> None:
    ordered = results_df.sort_values("agreement_score", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 7), dpi=180)
    bars = ax.barh(ordered["id"], ordered["agreement_score"] * 100, color="#7C3AED")
    ax.set_title("Explanation Agreement by Testcase", fontsize=16, fontweight="bold")
    ax.set_xlabel("Agreement Score (%)")
    ax.set_ylabel("Case ID")
    ax.set_xlim(0, 100)
    ax.axvline(20, color="#DC2626", linestyle="--", linewidth=1.2, label="Low-agreement flag")
    for bar, value in zip(bars, ordered["agreement_score"] * 100):
        ax.text(value + 1, bar.get_y() + bar.get_height() / 2, f"{value:.1f}%", va="center")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "05_explanation_agreement.png"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    backend_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    parser = argparse.ArgumentParser(description="Generate PPT-ready testcase report and charts.")
    parser.add_argument(
        "--cases",
        default=os.path.join(backend_root, "data", "ppt_demo_cases.json"),
        help="Path to testcase JSON file.",
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(backend_root, "reports", "ppt_demo"),
        help="Directory to save CSV, JSON, and PNG outputs.",
    )
    parser.add_argument(
        "--model_dir",
        default=None,
        help="Override model saved directory (default: backend/model/saved).",
    )
    args = parser.parse_args()

    _safe_mkdir(args.output_dir)
    spec = _load_cases(args.cases)

    device = get_device()
    model, tokenizer = load_model(args.model_dir)
    decision_threshold = load_decision_threshold(backend_root)

    rows = [
        _run_case(case, model, tokenizer, device, decision_threshold)
        for case in spec["cases"]
    ]
    results_df = pd.DataFrame(rows)
    results_df["pass"] = results_df["pass"].astype(bool)

    results_csv = os.path.join(args.output_dir, "case_results.csv")
    results_df.to_csv(results_csv, index=False)

    summary = _export_summary(
        results_df,
        {
            "description": spec.get("description", ""),
            "decision_threshold": decision_threshold,
        },
        args.output_dir,
    )
    category_df = pd.DataFrame(summary["category_summary"])
    category_df.to_csv(os.path.join(args.output_dir, "category_summary.csv"), index=False)

    _set_plot_defaults()
    _save_overall_chart(summary, args.output_dir)
    _save_category_chart(category_df, args.output_dir)
    _save_confusion_chart(results_df, args.output_dir)
    _save_probability_chart(results_df, decision_threshold, args.output_dir)
    _save_agreement_chart(results_df, args.output_dir)

    print(f"Saved PPT report outputs to: {args.output_dir}")
    print(f"Accuracy: {summary['accuracy'] * 100:.1f}% ({summary['passed_cases']}/{summary['total_cases']})")
    if summary["failed_case_ids"]:
        print(f"Failed case IDs: {', '.join(summary['failed_case_ids'])}")


if __name__ == "__main__":
    main()
