#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""
All-in-one stress classifier training for Google Colab (or any machine).

Does the same workflow as the project backend trainer:
  • pip installs (optional) • Dreaddit + bundled hard-negative seed (+ optional CSV/HF)
  • class-weighted fine-tuning • early stopping on val F1
  • saves model, tokenizer, training_metadata.json, threshold_analysis.json, run_summary.json
  • optional emotion_aux.joblib (TF-IDF + logistic diagnostics)

Colab usage (recommended):
  1. Runtime → Change runtime type → GPU
  2. Upload this file to /content/ OR clone your repo and open this path
  3. Run ONE cell:

     !python /content/colab_train_stress_all_in_one.py \\
         --output_dir /content/stress_model_saved \\
         --multi_task

  4. Download the folder or zip from the Files panel (see final print).

Local usage:
  python colab_train_stress_all_in_one.py --output_dir ./stress_saved
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import subprocess
import sys
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# ── embedded seed (same as backend/data/hard_negatives_seed.csv) ─────────────
HARD_NEGATIVES_SEED_CSV = """text,label
i am not stressed,0
i'm not stressed,0
im not stressed,0
i am not stressed at all,0
not stressed anymore,0
never stressed about it,0
no stress here,0
no stress today,0
zero stress,0
stress free day,0
having a stress free week,0
feeling stress free,0
i feel calm,0
i am calm,0
i am calm all the time,0
mostly calm lately,0
everything is fine,0
i'm okay,0
i am okay,0
doing okay,0
not feeling stressed,0
don't feel stressed,0
do not feel stressed,0
not very stressed,0
managing stress well,0
keeping stress under control,0
life is peaceful,0
pretty relaxed today,0
just chilling,0
all good here,0
nothing to worry about,0
i'm fine really,0
feeling neutral,0
not overwhelmed,0
not anxious right now,0
no anxiety today,0
handling work fine,0
coping well,0
sleeping fine,0
enjoying the weekend,0
looking forward to vacation,0
grateful and calm,0
not burned out,0
not burned out anymore,0
took a break and feel better,0
self care day went well,0
journal says i'm stable,0
therapist says i'm stable,0
mood is stable,0
low stress week,0
minimal stress,0
barely any stress,0
i'm not under pressure,0
deadlines are manageable,0
everything under control,0
i am relaxed,0
feeling relaxed,0
taking it easy,0
not a big deal,0
it's fine honestly,0
i am good,0
i'm alright,0
no complaints,0
smooth sailing,0
life feels normal,0
ordinary day,0
boring in a good way,0
not tense at all,0
muscles not tense,0
heart rate normal,0
breathing is steady,0
not ruminating,0
mind is clear,0
focused and calm,0
balanced mood,0
emotionally steady,0
neutral about work,0
neutral about school,0
short text ok test one,0
ok,0
k,0
sure,0
yep all good,0
nope not stressed,0
nah i'm fine,0
i am stressed,1
i'm so stressed,1
i am very stressed,1
completely overwhelmed,1
burning out at work,1
cannot sleep from stress,1
panic about deadlines,1
anxiety is crushing me,1
constant worry,1
i feel trapped,1
mental breakdown energy,1
too much pressure,1
deadlines are killing me,1
scared i will fail,1
heart racing from stress,1
chest tight from anxiety,1
ruminating every night,1
short i'm overwhelmed,1
stressed,1
"""

TextLabelPairs = List[Tuple[str, int]]


def default_output_dir() -> str:
    if os.path.isdir("/content"):
        return "/content/stress_model_saved"
    return os.path.join(os.getcwd(), "stress_model_saved")


def ensure_dependencies(quiet: bool = True) -> None:
    pkgs = [
        "torch",
        "transformers>=4.36.0",
        "datasets>=2.16.0",
        "accelerate",
        "safetensors",
        "scikit-learn",
        "pandas",
        "joblib",
        "tqdm",
    ]
    cmd = [sys.executable, "-m", "pip", "install"]
    if quiet:
        cmd.append("-q")
    cmd.extend(pkgs)
    print("Installing / verifying dependencies:", " ".join(pkgs))
    subprocess.check_call(cmd)


def clean_text(text: str) -> str:
    text = str(text).strip()
    if not text:
        return ""
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"^RT[\s:]+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#", " ", text)
    text = re.sub(r"\[removed\]|\[deleted\]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[^\w\s'\-.,!?]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _normalize_label(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        iv = int(value)
        if iv in (0, 1):
            return iv
    if isinstance(value, str):
        v = value.strip().lower().replace("-", " ").replace("_", " ")
        if v in (
            "1",
            "true",
            "yes",
            "y",
            "stress",
            "stressed",
            "stress positive",
            "positive",
            "pos",
        ):
            return 1
        if v in (
            "0",
            "false",
            "no",
            "n",
            "not stressed",
            "non stressed",
            "stress negative",
            "negative",
            "neg",
            "normal",
        ):
            return 0
    return None


def _dedupe(rows: Sequence[Tuple[str, int]]) -> TextLabelPairs:
    seen, out = set(), []
    for text, label in rows:
        key = (text.lower(), int(label))
        if key not in seen:
            seen.add(key)
            out.append((text, int(label)))
    return out


def load_seed_embedded() -> TextLabelPairs:
    rows: TextLabelPairs = []
    reader = csv.DictReader(io.StringIO(HARD_NEGATIVES_SEED_CSV))
    for row in reader:
        t = clean_text(row["text"])
        if not t:
            continue
        rows.append((t, int(row["label"])))
    return _dedupe(rows)


def load_dreaddit() -> TextLabelPairs:
    from datasets import load_dataset

    dataset = None
    err: List[str] = []
    for candidate in ("andreagasparini/dreaddit", "dreaddit"):
        try:
            dataset = load_dataset(candidate)
            print(f"  Loaded Dreaddit: {candidate}")
            break
        except Exception as exc:
            err.append(f"{candidate}: {exc}")
    if dataset is None:
        raise RuntimeError("Could not load Dreaddit: " + " | ".join(err))

    rows: TextLabelPairs = []
    for split_name in dataset.keys():
        split = dataset[split_name]
        for text, label in zip(split["text"], split["label"]):
            t = clean_text(str(text))
            if t:
                rows.append((t, int(label)))
    return _dedupe(rows)


def load_local_csv(path: str, text_col: str, label_col: str) -> TextLabelPairs:
    import pandas as pd

    df = pd.read_csv(path)
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"{path} needs columns {text_col}, {label_col}; got {list(df.columns)}")
    rows: TextLabelPairs = []
    for _, row in df.iterrows():
        t = clean_text(str(row[text_col]))
        if not t:
            continue
        lab = _normalize_label(row[label_col])
        if lab is None:
            continue
        rows.append((t, lab))
    return _dedupe(rows)


def load_hf_extra(
    dataset_name: str,
    text_col: str = "text",
    label_col: str = "label",
    split_names: Iterable[str] = ("train", "validation", "test"),
) -> TextLabelPairs:
    from datasets import load_dataset

    ds = load_dataset(dataset_name)
    rows: TextLabelPairs = []
    for sn in split_names:
        if sn not in ds:
            continue
        sp = ds[sn]
        for text, label in zip(sp[text_col], sp[label_col]):
            t = clean_text(str(text))
            if not t:
                continue
            lab = _normalize_label(label)
            if lab is None:
                continue
            rows.append((t, lab))
    return _dedupe(rows)


def summarize_corpus(corpus: TextLabelPairs) -> Dict[str, Any]:
    labels = [l for _, l in corpus]
    c = Counter(labels)
    lens = [len(t.split()) for t, _ in corpus]
    return {
        "num_rows": len(corpus),
        "label_counts": {"0": int(c.get(0, 0)), "1": int(c.get(1, 0))},
        "avg_tokens": round(sum(lens) / max(1, len(lens)), 2),
    }


# ── emotion aux (matches backend/model/emotion_aux.py) ────────────────────────

EMOTION_LABELS = ["anxiety", "burnout", "pressure", "distress", "none"]
_EMOTION_LEXICON = {
    "anxiety": {"anxiety", "anxious", "panic", "nervous", "worried", "fear"},
    "burnout": {"burnout", "burned", "exhausted", "drained", "fatigued", "tired"},
    "pressure": {"deadline", "deadlines", "pressure", "burden", "responsibility"},
    "distress": {"helpless", "hopeless", "trapped", "breakdown", "overwhelmed", "stuck"},
}


def _derive_emotion_label(text: str) -> str:
    words = [w.strip(".,!?;:\"'()[]{}").lower() for w in text.split()]
    words = [w for w in words if w]
    if not words:
        return "none"
    counts = {k: 0 for k in _EMOTION_LEXICON}
    for w in words:
        for emotion, lex in _EMOTION_LEXICON.items():
            if w in lex:
                counts[emotion] += 1
    if sum(counts.values()) == 0:
        return "none"
    return max(counts.items(), key=lambda x: x[1])[0]


def train_and_save_emotion_aux(train_texts: List[str], save_dir: str) -> None:
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    labels = [_derive_emotion_label(t) for t in train_texts]
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    x = vectorizer.fit_transform(train_texts)
    clf = LogisticRegression(max_iter=500)
    clf.fit(x, labels)
    bundle = {"vectorizer": vectorizer, "classifier": clf, "labels": EMOTION_LABELS}
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(bundle, os.path.join(save_dir, "emotion_aux.joblib"))
    print("Saved emotion_aux.joblib")


# ── training ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Colab all-in-one stress trainer")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=default_output_dir(),
        help="Where to save the fine-tuned model (default: /content/... on Colab, else ./stress_model_saved)",
    )
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--early_stopping_patience", type=int, default=2)
    parser.add_argument("--pretrained_model", type=str, default="bert-base-uncased")
    parser.add_argument("--no_dreaddit", action="store_true")
    parser.add_argument("--no_hard_negative_seed", action="store_true")
    parser.add_argument("--multi_task", action="store_true", help="Save emotion_aux.joblib")
    parser.add_argument(
        "--csv_paths",
        nargs="*",
        default=[],
        help="Extra CSVs with text,label (override cols with --text_col / --label_col)",
    )
    parser.add_argument("--text_col", type=str, default="text")
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument(
        "--extra_hf_datasets",
        nargs="*",
        default=[],
        help="Additional HF dataset ids with text+label columns",
    )
    parser.add_argument("--skip_pip", action="store_true", help="Do not run pip install")
    parser.add_argument("--make_zip", action="store_true", help="Write output_dir.zip")
    args = parser.parse_args()

    if not args.skip_pip:
        ensure_dependencies()

    import torch
    import torch.nn.functional as F
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from transformers import get_linear_schedule_with_warmup

    # build corpus
    corpus: TextLabelPairs = []
    source_counts: Dict[str, int] = {}

    if not args.no_hard_negative_seed:
        seed = load_seed_embedded()
        corpus.extend(seed)
        source_counts["hard_negatives_seed"] = len(seed)
        print(f"Hard-negative seed: {len(seed)} examples")

    if not args.no_dreaddit:
        print("Loading Dreaddit...")
        dres = load_dreaddit()
        corpus.extend(dres)
        source_counts["dreaddit"] = len(dres)
        print(f"  Dreaddit total after merge: {len(dres)}")

    for p in args.csv_paths:
        print(f"Loading CSV {p}")
        rows = load_local_csv(p, args.text_col, args.label_col)
        corpus.extend(rows)
        source_counts[f"csv:{os.path.basename(p)}"] = len(rows)

    for hf_id in args.extra_hf_datasets:
        print(f"Loading HF {hf_id}")
        rows = load_hf_extra(hf_id, args.text_col, args.label_col)
        corpus.extend(rows)
        source_counts[f"hf:{hf_id}"] = len(rows)

    corpus = _dedupe(corpus)
    print(f"Total merged: {len(corpus)} | {summarize_corpus(corpus)}")

    if len(corpus) < 200:
        raise SystemExit("Not enough data. Enable Dreaddit or add CSV/HF sources.")

    texts = [t for t, _ in corpus]
    labels = [int(l) for _, l in corpus]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Backbone: {args.pretrained_model}")

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.1, random_state=42, stratify=labels
    )
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=0.1111, random_state=42, stratify=train_labels
    )

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.pretrained_model, num_labels=2
    )
    model.to(device)

    class StressDataset(Dataset):
        def __init__(self, tx, lb, max_length=128):
            enc = tokenizer(
                tx,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
            self.input_ids = enc["input_ids"]
            self.attention_mask = enc["attention_mask"]
            self.labels = torch.tensor(lb, dtype=torch.long)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return {
                "input_ids": self.input_ids[idx],
                "attention_mask": self.attention_mask[idx],
                "labels": self.labels[idx],
            }

    def evaluate(m, loader):
        m.eval()
        all_preds, all_labels_, all_probs = [], [], []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = m(**batch)
                probs = torch.softmax(out.logits, dim=-1)[:, 1]
                preds = torch.argmax(out.logits, dim=-1)
                all_probs.extend(probs.cpu().numpy().tolist())
                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels_.extend(batch["labels"].cpu().numpy().tolist())
        met = {
            "accuracy": accuracy_score(all_labels_, all_preds),
            "f1": f1_score(all_labels_, all_preds, zero_division=0),
            "precision": precision_score(all_labels_, all_preds, zero_division=0),
            "recall": recall_score(all_labels_, all_preds, zero_division=0),
        }
        try:
            met["roc_auc"] = roc_auc_score(all_labels_, all_probs)
        except ValueError:
            met["roc_auc"] = 0.0
        return met, all_labels_, all_preds, all_probs

    train_loader = DataLoader(
        StressDataset(train_texts, train_labels), batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(StressDataset(val_texts, val_labels), batch_size=args.batch_size)
    test_loader = DataLoader(StressDataset(test_texts, test_labels), batch_size=args.batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    label_tensor = torch.tensor(train_labels, dtype=torch.long)
    class_counts = torch.bincount(label_tensor, minlength=2).float()
    class_weights = (class_counts.sum() / (2.0 * class_counts.clamp(min=1.0))).to(device)

    save_dir = args.output_dir
    os.makedirs(save_dir, exist_ok=True)

    best_val_f1 = 0.0
    stall = 0
    corpus_stats = {"source_counts": source_counts, "merged_summary": summarize_corpus(corpus)}

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            loss = F.cross_entropy(out.logits, batch["labels"], weight=class_weights)
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        val_m, _, _, _ = evaluate(model, val_loader)
        print(
            f"Epoch {epoch + 1}/{args.epochs} | loss={total_loss/len(train_loader):.4f} | "
            f"val_f1={val_m['f1']:.4f} val_acc={val_m['accuracy']:.4f}"
        )
        if val_m["f1"] >= best_val_f1:
            best_val_f1 = val_m["f1"]
            stall = 0
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            meta = {
                "train_script": "colab_train_stress_all_in_one.py",
                "pretrained_model": args.pretrained_model,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "early_stopping_patience": args.early_stopping_patience,
                "train_size": len(train_texts),
                "val_size": len(val_texts),
                "test_size": len(test_texts),
                "use_dreaddit": not args.no_dreaddit,
                "include_hard_negative_seed": not args.no_hard_negative_seed,
                "csv_paths": args.csv_paths,
                "extra_hf_datasets": args.extra_hf_datasets,
                "text_col": args.text_col,
                "label_col": args.label_col,
                "multi_task": args.multi_task,
                "corpus_stats": corpus_stats,
            }
            with open(os.path.join(save_dir, "training_metadata.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            print(f"  -> checkpoint saved (val_f1={val_m['f1']:.4f})")
        else:
            stall += 1
            if stall >= args.early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    model = AutoModelForSequenceClassification.from_pretrained(save_dir, num_labels=2)
    model.to(device)
    test_m, y_true, y_pred, y_prob = evaluate(model, test_loader)

    thresholds = (0.3, 0.4, 0.5, 0.6, 0.7)
    threshold_report = []
    for t in thresholds:
        yhat = [1 if p >= t else 0 for p in y_prob]
        threshold_report.append(
            {
                "threshold": round(t, 2),
                "precision": round(precision_score(y_true, yhat, zero_division=0), 4),
                "recall": round(recall_score(y_true, yhat, zero_division=0), 4),
                "f1": round(f1_score(y_true, yhat, zero_division=0), 4),
                "accuracy": round(accuracy_score(y_true, yhat), 4),
            }
        )
    best_thr_row = max(threshold_report, key=lambda r: r["f1"])
    with open(os.path.join(save_dir, "threshold_analysis.json"), "w", encoding="utf-8") as f:
        json.dump(threshold_report, f, indent=2)

    run_summary = {
        "best_val_f1": round(best_val_f1, 4),
        "test_accuracy": round(float(test_m["accuracy"]), 4),
        "test_f1": round(float(test_m["f1"]), 4),
        "test_precision": round(float(test_m["precision"]), 4),
        "test_recall": round(float(test_m["recall"]), 4),
        "test_roc_auc": round(float(test_m["roc_auc"]), 4),
        "recommended_threshold": float(best_thr_row["threshold"]),
    }
    with open(os.path.join(save_dir, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)

    print("\n=== Test metrics ===")
    print(
        classification_report(y_true, y_pred, target_names=["Not Stressed", "Stressed"])
    )
    print("run_summary:", json.dumps(run_summary, indent=2))

    if args.multi_task:
        train_and_save_emotion_aux(train_texts, save_dir)

    if args.make_zip:
        import shutil

        base = os.path.abspath(save_dir)
        parent = os.path.dirname(base)
        name = os.path.basename(base)
        arch = shutil.make_archive(base, "zip", root_dir=parent, base_dir=name)
        print("Created archive:", arch)

    print(
        "\nDone. Copy everything under:\n  ",
        os.path.abspath(save_dir),
        "\ninto your project's backend/model/saved/ for local inference.",
        sep="",
    )


if __name__ == "__main__":
    main()
