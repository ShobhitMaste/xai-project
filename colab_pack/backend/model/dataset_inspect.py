"""
Dataset inspection utility for stress training inputs.

Usage:
    cd backend
    python model/dataset_inspect.py --csv_paths ../resources/dreaddit-train.csv
"""

import argparse
import json
import os
import sys

try:
    from model.data_loader import build_training_corpus, inspect_local_csv
except ModuleNotFoundError:
    # Support direct execution: `python model/dataset_inspect.py`
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from model.data_loader import build_training_corpus, inspect_local_csv


def main():
    parser = argparse.ArgumentParser(description="Inspect local/HF stress datasets.")
    parser.add_argument("--no_dreaddit", action="store_true")
    parser.add_argument("--csv_paths", nargs="*", default=[])
    parser.add_argument("--extra_hf_datasets", nargs="*", default=[])
    parser.add_argument("--text_col", type=str, default="text")
    parser.add_argument("--label_col", type=str, default="label")
    args = parser.parse_args()

    print("=== Per-CSV Inspection ===")
    if not args.csv_paths:
        print("No local CSV paths provided.")
    for path in args.csv_paths:
        report = inspect_local_csv(path, text_col=args.text_col, label_col=args.label_col)
        print(json.dumps(report, indent=2))

    print("\n=== Merged Corpus Inspection ===")
    corpus, stats = build_training_corpus(
        use_dreaddit=not args.no_dreaddit,
        local_csv_paths=args.csv_paths,
        extra_hf_datasets=args.extra_hf_datasets,
        text_col=args.text_col,
        label_col=args.label_col,
        return_stats=True,
    )
    print(f"Merged rows: {len(corpus)}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
