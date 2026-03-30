import os
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import pandas as pd
from datasets import load_dataset

from utils.preprocessing import clean_text

_BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HARD_NEGATIVES_SEED_CSV = os.path.join(_BACKEND_ROOT, "data", "hard_negatives_seed.csv")

TextLabelPairs = List[Tuple[str, int]]


def _normalize_label(value) -> Optional[int]:
    """Map common binary representations into {0,1}; return None if unknown."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        iv = int(value)
        if iv in (0, 1):
            return iv
        return None
    if isinstance(value, str):
        v = value.strip().lower()
        v = v.replace("-", " ").replace("_", " ")
        positives = {
            "1",
            "true",
            "yes",
            "y",
            "stress",
            "stressed",
            "stress positive",
            "positive",
            "pos",
        }
        negatives = {
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
        }
        if v in positives:
            return 1
        if v in negatives:
            return 0
    return None


def _dedupe_examples(rows: Sequence[Tuple[str, int]]) -> TextLabelPairs:
    seen = set()
    out: TextLabelPairs = []
    for text, label in rows:
        key = (text.lower(), int(label))
        if key not in seen:
            seen.add(key)
            out.append((text, int(label)))
    return out


def summarize_rows(rows: Sequence[Tuple[str, int]]) -> Dict[str, object]:
    texts = [t for t, _ in rows]
    labels = [int(l) for _, l in rows]
    label_counts = Counter(labels)
    duplicates = len(texts) - len(set([t.lower() for t in texts]))
    lengths = [len(t.split()) for t in texts] if texts else [0]
    return {
        "num_rows": len(rows),
        "label_counts": {"0": int(label_counts.get(0, 0)), "1": int(label_counts.get(1, 0))},
        "duplicates_by_text": int(duplicates),
        "avg_tokens": round(sum(lengths) / max(1, len(lengths)), 2),
        "min_tokens": int(min(lengths)) if lengths else 0,
        "max_tokens": int(max(lengths)) if lengths else 0,
    }


def load_dreaddit() -> TextLabelPairs:
    """
    Load Dreaddit from HuggingFace.
    Expected columns: text, label (0/1).
    """
    # Some environments resolve Dreaddit only via explicit namespace.
    dataset = None
    load_errors = []
    for candidate in ("andreagasparini/dreaddit", "dreaddit"):
        try:
            dataset = load_dataset(candidate)
            print(f"  Loaded Dreaddit source: {candidate}")
            break
        except Exception as exc:
            load_errors.append(f"{candidate}: {exc}")
    if dataset is None:
        joined = " | ".join(load_errors)
        raise RuntimeError(
            "Could not load Dreaddit from HuggingFace. "
            "Tried candidates: andreagasparini/dreaddit, dreaddit. "
            f"Errors: {joined}"
        )
    rows: TextLabelPairs = []
    for split_name in dataset.keys():
        split = dataset[split_name]
        for text, label in zip(split["text"], split["label"]):
            text = clean_text(str(text))
            if text:
                rows.append((text, int(label)))
    return _dedupe_examples(rows)


def load_local_csv(
    csv_path: str, text_col: str = "text", label_col: str = "label"
) -> TextLabelPairs:
    """Load one local CSV with text+label columns."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(
            f"{csv_path} must contain columns '{text_col}' and '{label_col}'. "
            f"Found: {list(df.columns)}"
        )

    rows: TextLabelPairs = []
    skipped_unknown_labels = 0
    for _, row in df.iterrows():
        text = clean_text(str(row[text_col]))
        if not text:
            continue
        label = _normalize_label(row[label_col])
        if label is None:
            skipped_unknown_labels += 1
            continue
        rows.append((text, label))
    if skipped_unknown_labels:
        print(
            f"  Warning: skipped {skipped_unknown_labels} rows with unknown labels in {csv_path}"
        )
    return _dedupe_examples(rows)


def inspect_local_csv(
    csv_path: str, text_col: str = "text", label_col: str = "label"
) -> Dict[str, object]:
    """Inspect local CSV and return quality statistics."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(
            f"{csv_path} must contain columns '{text_col}' and '{label_col}'. "
            f"Found: {list(df.columns)}"
        )

    raw_label_values = df[label_col].dropna().astype(str).str.strip().str.lower().tolist()
    raw_counts = Counter(raw_label_values)

    cleaned_rows: TextLabelPairs = []
    empty_text_rows = 0
    unknown_label_rows = 0
    for _, row in df.iterrows():
        text = clean_text(str(row[text_col]))
        if not text:
            empty_text_rows += 1
            continue
        label = _normalize_label(row[label_col])
        if label is None:
            unknown_label_rows += 1
            continue
        cleaned_rows.append((text, label))

    summary = summarize_rows(cleaned_rows)
    return {
        "path": csv_path,
        "raw_rows": int(len(df)),
        "empty_text_rows": int(empty_text_rows),
        "unknown_label_rows": int(unknown_label_rows),
        "raw_label_values": dict(raw_counts),
        "clean_summary": summary,
    }


def load_hf_binary_dataset(
    dataset_name: str,
    text_col: str = "text",
    label_col: str = "label",
    split_names: Iterable[str] = ("train", "validation", "test"),
) -> TextLabelPairs:
    """
    Generic loader for binary HuggingFace datasets.
    Use this for additional stress datasets with text/label columns.
    """
    ds = load_dataset(dataset_name)
    rows: TextLabelPairs = []
    for split_name in split_names:
        if split_name not in ds:
            continue
        split = ds[split_name]
        if text_col not in split.column_names or label_col not in split.column_names:
            raise ValueError(
                f"{dataset_name}:{split_name} missing '{text_col}'/'{label_col}'. "
                f"Columns: {split.column_names}"
            )
        for text, label in zip(split[text_col], split[label_col]):
            cleaned = clean_text(str(text))
            if not cleaned:
                continue
            normalized = _normalize_label(label)
            if normalized is None:
                continue
            rows.append((cleaned, normalized))
    return _dedupe_examples(rows)


def build_training_corpus(
    use_dreaddit: bool = True,
    local_csv_paths: Sequence[str] = (),
    extra_hf_datasets: Sequence[str] = (),
    text_col: str = "text",
    label_col: str = "label",
    return_stats: bool = False,
    include_hard_negative_seed: bool = True,
) -> Union[TextLabelPairs, Tuple[TextLabelPairs, Dict[str, object]]]:
    """Aggregate many datasets into one deduplicated corpus."""
    corpus: TextLabelPairs = []
    source_counts: Dict[str, int] = {}

    if include_hard_negative_seed and os.path.isfile(HARD_NEGATIVES_SEED_CSV):
        print(f"Loading bundled hard-negative seed: {HARD_NEGATIVES_SEED_CSV}")
        # Seed file always uses default column names independent of --text_col/--label_col.
        seed_rows = load_local_csv(HARD_NEGATIVES_SEED_CSV, text_col="text", label_col="label")
        print(f"  Hard-negative seed examples: {len(seed_rows)}")
        corpus.extend(seed_rows)
        source_counts["hard_negatives_seed"] = len(seed_rows)

    if use_dreaddit:
        print("Loading Dreaddit...")
        dreaddit_rows = load_dreaddit()
        print(f"  Dreaddit examples: {len(dreaddit_rows)}")
        corpus.extend(dreaddit_rows)
        source_counts["dreaddit"] = len(dreaddit_rows)

    for csv_path in local_csv_paths:
        print(f"Loading CSV: {csv_path}")
        csv_rows = load_local_csv(csv_path, text_col=text_col, label_col=label_col)
        print(f"  CSV examples: {len(csv_rows)}")
        corpus.extend(csv_rows)
        source_counts[f"csv:{os.path.basename(csv_path)}"] = len(csv_rows)

    for dataset_name in extra_hf_datasets:
        print(f"Loading HF dataset: {dataset_name}")
        hf_rows = load_hf_binary_dataset(
            dataset_name, text_col=text_col, label_col=label_col
        )
        print(f"  HF examples: {len(hf_rows)}")
        corpus.extend(hf_rows)
        source_counts[f"hf:{dataset_name}"] = len(hf_rows)

    corpus = _dedupe_examples(corpus)
    print(f"Total merged examples: {len(corpus)}")
    print(f"Merged label distribution: {summarize_rows(corpus)['label_counts']}")
    if len(corpus) < 1000:
        print(
            "Warning: merged corpus is still small. Add more CSV/HF datasets for a "
            "stronger production model."
        )
    if return_stats:
        return corpus, {
            "source_counts": source_counts,
            "merged_summary": summarize_rows(corpus),
        }
    return corpus
