"""
Fine-tune a transformer encoder for binary stress detection from multiple datasets.

Usage examples:
    cd backend
    python model/train.py
    python model/train.py --csv_paths data/stress_a.csv data/stress_b.csv
    python model/train.py --extra_hf_datasets my-org/stress-dataset
    python model/train.py --pretrained_model roberta-base
    python model/train.py --no_hard_negative_seed
"""

import argparse
import json
import os
import sys

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
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

try:
    from model.data_loader import build_training_corpus
    from model.emotion_aux import save_emotion_aux, train_emotion_aux
except ModuleNotFoundError:
    # Support direct execution: `python model/train.py`
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from model.data_loader import build_training_corpus
    from model.emotion_aux import save_emotion_aux, train_emotion_aux


class StressDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


def _compute_metrics(y_true, y_pred, y_prob):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }
    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics["roc_auc"] = 0.0
    return metrics


def _evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            probs = torch.softmax(outputs.logits, dim=-1)[:, 1]
            preds = torch.argmax(outputs.logits, dim=-1)
            all_probs.extend(probs.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(batch["labels"].cpu().numpy().tolist())
    return _compute_metrics(all_labels, all_preds, all_probs), all_labels, all_preds, all_probs


def _threshold_analysis(y_true, y_prob, thresholds=(0.3, 0.4, 0.5, 0.6, 0.7)):
    report = []
    for t in thresholds:
        y_pred = [1 if p >= t else 0 for p in y_prob]
        report.append(
            {
                "threshold": round(t, 2),
                "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
                "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
                "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
                "accuracy": round(accuracy_score(y_true, y_pred), 4),
            }
        )
    return report


def train(
    epochs: int = 4,
    batch_size: int = 16,
    lr: float = 2e-5,
    early_stopping_patience: int = 2,
    save_dir: str = None,
    use_dreaddit: bool = True,
    csv_paths=None,
    extra_hf_datasets=None,
    text_col: str = "text",
    label_col: str = "label",
    multi_task: bool = False,
    pretrained_model: str = "bert-base-uncased",
    include_hard_negative_seed: bool = True,
):
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved")
    csv_paths = csv_paths or []
    extra_hf_datasets = extra_hf_datasets or []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    corpus, corpus_stats = build_training_corpus(
        use_dreaddit=use_dreaddit,
        local_csv_paths=csv_paths,
        extra_hf_datasets=extra_hf_datasets,
        text_col=text_col,
        label_col=label_col,
        return_stats=True,
        include_hard_negative_seed=include_hard_negative_seed,
    )
    if len(corpus) < 200:
        raise ValueError(
            "Not enough training data loaded. Provide more CSV/HF datasets and retry."
        )

    texts = [t for t, _ in corpus]
    labels = [int(l) for _, l in corpus]

    print(f"Pretrained backbone: {pretrained_model}")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model, num_labels=2
    )
    model.to(device)
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.1, random_state=42, stratify=labels
    )
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=0.1111, random_state=42, stratify=train_labels
    )  # ~80/10/10 train/val/test

    train_dataset = StressDataset(train_texts, train_labels, tokenizer)
    val_dataset = StressDataset(val_texts, val_labels, tokenizer)
    test_dataset = StressDataset(test_texts, test_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    # Resource-inspired improvement: weighted loss for class imbalance.
    label_tensor = torch.tensor(train_labels, dtype=torch.long)
    class_counts = torch.bincount(label_tensor, minlength=2).float()
    class_weights = (class_counts.sum() / (2.0 * class_counts.clamp(min=1.0))).to(device)
    print(
        "Class distribution (train): "
        f"not_stressed={int(class_counts[0].item())}, stressed={int(class_counts[1].item())}"
    )
    print(
        "Class weights: "
        f"not_stressed={class_weights[0].item():.4f}, stressed={class_weights[1].item():.4f}"
    )

    best_val_f1 = 0.0
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels_batch = batch["labels"]
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            loss = F.cross_entropy(outputs.logits, labels_batch, weight=class_weights)
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        avg_loss = total_loss / len(train_loader)

        val_metrics, _, _, _ = _evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | Val F1: {val_metrics['f1']:.4f} | "
            f"Val Precision: {val_metrics['precision']:.4f} | Val Recall: {val_metrics['recall']:.4f} | "
            f"Val ROC-AUC: {val_metrics['roc_auc']:.4f}"
        )

        if val_metrics["f1"] >= best_val_f1:
            best_val_f1 = val_metrics["f1"]
            epochs_without_improvement = 0
            os.makedirs(save_dir, exist_ok=True)
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            metadata = {
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": lr,
                "early_stopping_patience": early_stopping_patience,
                "train_size": len(train_texts),
                "val_size": len(val_texts),
                "test_size": len(test_texts),
                "use_dreaddit": use_dreaddit,
                "csv_paths": csv_paths,
                "extra_hf_datasets": extra_hf_datasets,
                "text_col": text_col,
                "label_col": label_col,
                "multi_task": multi_task,
                "corpus_stats": corpus_stats,
                "pretrained_model": pretrained_model,
                "include_hard_negative_seed": include_hard_negative_seed,
            }
            with open(os.path.join(save_dir, "training_metadata.json"), "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
            print(f"  -> Saved best model (val_f1={val_metrics['f1']:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                print(
                    f"Early stopping triggered after {epoch + 1} epochs "
                    f"(patience={early_stopping_patience})."
                )
                break

    # Load best checkpoint for final test report
    model = AutoModelForSequenceClassification.from_pretrained(save_dir, num_labels=2)
    model.to(device)
    test_metrics, test_labels_true, test_preds, test_probs = _evaluate(model, test_loader, device)
    threshold_report = _threshold_analysis(test_labels_true, test_probs)

    print(f"\nTraining complete. Best validation F1: {best_val_f1:.4f}")
    print(
        f"Final Test Metrics | Acc: {test_metrics['accuracy']:.4f} | "
        f"F1: {test_metrics['f1']:.4f} | Precision: {test_metrics['precision']:.4f} | "
        f"Recall: {test_metrics['recall']:.4f} | ROC-AUC: {test_metrics['roc_auc']:.4f}"
    )
    print("\nFinal Test Classification Report:")
    print(
        classification_report(
            test_labels_true, test_preds, target_names=["Not Stressed", "Stressed"]
        )
    )
    print("\nThreshold Analysis (test set):")
    for row in threshold_report:
        print(
            f"  thr={row['threshold']:.2f} | acc={row['accuracy']:.4f} | "
            f"precision={row['precision']:.4f} | recall={row['recall']:.4f} | f1={row['f1']:.4f}"
        )

    threshold_path = os.path.join(save_dir, "threshold_analysis.json")
    with open(threshold_path, "w", encoding="utf-8") as f:
        json.dump(threshold_report, f, indent=2)
    print(f"Saved threshold analysis to: {threshold_path}")

    if multi_task:
        print("Training auxiliary emotion diagnostic model...")
        emotion_bundle = train_emotion_aux(train_texts)
        save_emotion_aux(emotion_bundle, save_dir)
        print("Saved auxiliary emotion model: emotion_aux.joblib")

    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train stress detection model on real datasets.")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--early_stopping_patience", type=int, default=2)
    parser.add_argument("--no_dreaddit", action="store_true", help="Disable Dreaddit loading.")
    parser.add_argument(
        "--csv_paths",
        nargs="*",
        default=[],
        help="Local CSV files with text/label columns (space-separated).",
    )
    parser.add_argument(
        "--extra_hf_datasets",
        nargs="*",
        default=[],
        help="Additional HuggingFace dataset IDs with binary labels.",
    )
    parser.add_argument("--text_col", type=str, default="text")
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument(
        "--multi_task",
        action="store_true",
        help="Enable auxiliary emotion-task training and save emotion diagnostics model.",
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default="bert-base-uncased",
        help="HF model id (e.g. bert-base-uncased, roberta-base).",
    )
    parser.add_argument(
        "--no_hard_negative_seed",
        action="store_true",
        help="Do not merge bundled data/hard_negatives_seed.csv.",
    )
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        early_stopping_patience=args.early_stopping_patience,
        use_dreaddit=not args.no_dreaddit,
        csv_paths=args.csv_paths,
        extra_hf_datasets=args.extra_hf_datasets,
        text_col=args.text_col,
        label_col=args.label_col,
        multi_task=args.multi_task,
        pretrained_model=args.pretrained_model,
        include_hard_negative_seed=not args.no_hard_negative_seed,
    )
