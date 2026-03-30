"""
Evaluate hand-written cases (negation, short calm, obvious stress) using the same
guardrails as production, without running SHAP/LIME (fast).

Usage (from backend/):
    python model/eval_golden.py
    python model/eval_golden.py --golden data/golden_eval.json --min-accuracy 0.8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

try:
    from model.explain import get_emotion_signals
    from model.guardrails import apply_conservative_postcheck
    from model.model_loader import get_device, load_model
    from model.predict import predict
    from utils.preprocessing import clean_text
except ModuleNotFoundError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from model.explain import get_emotion_signals
    from model.guardrails import apply_conservative_postcheck
    from model.model_loader import get_device, load_model
    from model.predict import predict
    from utils.preprocessing import clean_text


def load_decision_threshold(backend_root: str) -> float:
    path = os.path.join(backend_root, "model", "saved", "threshold_analysis.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            rows = json.load(f)
        if isinstance(rows, list) and rows:
            best = max(rows, key=lambda r: float(r.get("f1", 0.0)))
            return float(best.get("threshold", 0.5))
    except Exception:
        pass
    return 0.5


def final_label_like_api(
    raw_text: str,
    model,
    tokenizer,
    device,
    decision_threshold: float,
) -> str:
    cleaned = clean_text(raw_text)
    prediction = predict(
        cleaned,
        model,
        tokenizer,
        device,
        decision_threshold=decision_threshold,
    )
    minimal_explanations = {"emotion_signals": get_emotion_signals(cleaned)}
    out = apply_conservative_postcheck(cleaned, prediction, minimal_explanations)
    return out["label"]


def run_golden_eval(
    golden_path: str | None = None,
    model_dir: str | None = None,
) -> Tuple[float, List[Dict[str, Any]], int]:
    backend_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    path = golden_path or os.path.join(backend_root, "data", "golden_eval.json")
    with open(path, "r", encoding="utf-8") as f:
        spec = json.load(f)

    cases: List[dict] = spec["cases"]
    device = get_device()
    model, tokenizer = load_model(model_dir)

    thr = load_decision_threshold(backend_root)
    failures: List[Dict[str, Any]] = []
    for case in cases:
        cid = case.get("id", "?")
        text = case["text"]
        expected = case["expected"]
        accept = case.get("accept")
        got = final_label_like_api(text, model, tokenizer, device, thr)
        ok = got in accept if accept else got == expected
        if not ok:
            failures.append({"id": cid, "text": text, "expected": expected, "got": got, "accept": accept})

    accuracy = (len(cases) - len(failures)) / max(1, len(cases))
    return accuracy, failures, len(cases)


def main():
    backend_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    parser = argparse.ArgumentParser(description="Golden-set slice evaluation (fast path).")
    parser.add_argument(
        "--golden",
        default=os.path.join(backend_root, "data", "golden_eval.json"),
        help="Path to golden_eval.json",
    )
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=0.0,
        help="Exit non-zero if accuracy is below this (0 disables).",
    )
    parser.add_argument(
        "--model_dir",
        default=None,
        help="Override model saved directory (default: model/saved).",
    )
    args = parser.parse_args()

    acc, failures, n = run_golden_eval(
        golden_path=args.golden,
        model_dir=args.model_dir,
    )
    print(f"Golden cases: {n} | accuracy: {acc:.3f}")
    if args.min_accuracy > 0 and acc + 1e-9 < args.min_accuracy:
        print(
            f"ERROR: accuracy {acc:.3f} below --min-accuracy {args.min_accuracy:.3f} "
            f"({len(failures)} failures)."
        )
        sys.exit(1)
    if failures:
        print("Failures:")
        for row in failures:
            print(f"  {row}")


if __name__ == "__main__":
    main()
