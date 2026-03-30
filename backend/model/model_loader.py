import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from model.emotion_aux import load_emotion_aux

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_dir: str = None):
    """Load the fine-tuned classifier and tokenizer from disk (BERT, RoBERTa, etc.)."""
    path = model_dir or MODEL_DIR
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found at {path}. Run 'python model/train.py' first."
        )
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        path, num_labels=2, output_attentions=True
    )
    model.emotion_aux = load_emotion_aux(path)
    model.to(DEVICE)
    model.eval()
    return model, tokenizer


def get_device():
    return DEVICE
