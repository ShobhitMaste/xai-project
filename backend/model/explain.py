"""
Explainability module: SHAP, LIME, and Attention-based explanations.

Each function takes raw text + model artifacts and returns a dict mapping
words to their contribution scores.
"""

import torch
import torch.nn.functional as F
import numpy as np
from lime.lime_text import LimeTextExplainer
import re

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "we",
    "with",
}


def _predict_proba(texts, model, tokenizer, device):
    """Batch prediction function used by SHAP and LIME."""
    results = []
    model.eval()
    for text in texts:
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, padding=True, max_length=128
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        results.append(probs[0].cpu().numpy())
    return np.array(results)


# ── SHAP ────────────────────────────────────────────────────────────────────

def get_shap_explanation(text, model, tokenizer, device, num_features=10):
    """
    Partition-based SHAP explanation.
    Returns top contributing words with their SHAP values for the Stressed class.
    """
    import shap

    def predict_fn(texts):
        return _predict_proba(list(texts), model, tokenizer, device)

    masker = shap.maskers.Text(r"\W+")
    explainer = shap.Explainer(
        predict_fn, masker, output_names=["Not Stressed", "Stressed"]
    )
    shap_values = explainer([text], max_evals=100)

    # shap_values.values shape: (1, num_tokens, 2)  -> pick class 1 (Stressed)
    if len(shap_values.values.shape) == 3:
        values = shap_values.values[0, :, 1]
    else:
        values = shap_values.values[0]

    tokens = shap_values.data[0]

    word_scores = {}
    for token, score in zip(tokens, values):
        token = token.strip()
        if token:
            word_scores[token] = round(float(score), 4)

    sorted_scores = dict(
        sorted(word_scores.items(), key=lambda x: abs(x[1]), reverse=True)[
            :num_features
        ]
    )
    return sorted_scores


# ── LIME ────────────────────────────────────────────────────────────────────

def get_lime_explanation(text, model, tokenizer, device, num_features=10):
    """
    LIME explanation for a single text instance.
    Returns word-level feature weights.
    """

    def predict_fn(texts):
        return _predict_proba(list(texts), model, tokenizer, device)

    explainer = LimeTextExplainer(class_names=["Not Stressed", "Stressed"])
    exp = explainer.explain_instance(
        text, predict_fn, num_features=num_features, num_samples=200
    )
    return {word: round(weight, 4) for word, weight in exp.as_list()}


# ── ATTENTION ───────────────────────────────────────────────────────────────

def get_attention_explanation(text, model, tokenizer, device):
    """
    Extract [CLS]-token attention from the last BERT layer, averaged over heads,
    and map back to whole words (merging sub-word tokens).
    """
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=128
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # Last layer, average across heads, take [CLS] row
    last_layer = outputs.attentions[-1]  # (1, heads, seq, seq)
    cls_attention = last_layer[0, :, 0, :].mean(dim=0)  # (seq,)

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    scores = cls_attention.cpu().numpy()

    # Merge sub-word pieces back into whole words
    word_scores = {}
    current_word, current_score = "", 0.0

    for tok, sc in zip(tokens, scores):
        if tok in ("[CLS]", "[SEP]", "[PAD]"):
            continue
        if tok.startswith("##"):
            current_word += tok[2:]
            current_score = max(current_score, float(sc))
        else:
            if current_word:
                word_scores[current_word] = current_score
            current_word = tok
            current_score = float(sc)

    if current_word:
        word_scores[current_word] = current_score

    # Normalize to [0, 1]
    if word_scores:
        lo = min(word_scores.values())
        hi = max(word_scores.values())
        rng = hi - lo if hi != lo else 1.0
        word_scores = {w: round((s - lo) / rng, 4) for w, s in word_scores.items()}

    return word_scores


# ── AGREEMENT + EMOTION SIGNALS ─────────────────────────────────────────────

_STRESS_EMOTION_LEXICON = {
    "anxiety": {
        "anxiety",
        "anxious",
        "panic",
        "nervous",
        "worried",
        "overwhelmed",
        "fear",
    },
    "burnout": {"burnout", "burned", "exhausted", "drained", "fatigued", "tired"},
    "pressure": {"deadline", "deadlines", "pressure", "burden", "responsibility"},
    "distress": {"helpless", "hopeless", "trapped", "stuck", "breakdown"},
}


def _normalize_word(word: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", word.lower()).strip()


def _is_informative_word(word: str) -> bool:
    normalized = _normalize_word(word)
    return bool(normalized) and len(normalized) > 2 and normalized not in _STOPWORDS


def get_explanation_agreement(explanations: dict, top_k: int = 8) -> dict:
    """
    Compute agreement score across SHAP, LIME, and Attention top words.
    """
    shap_words = set(list(explanations.get("shap", {}).keys())[:top_k])
    lime_words = set([k for k, _ in list(explanations.get("lime", {}).items())[:top_k]])
    attention_items = sorted(
        explanations.get("attention", {}).items(), key=lambda x: x[1], reverse=True
    )
    attention_words = set([k for k, _ in attention_items[:top_k]])

    shap_words = {_normalize_word(w) for w in shap_words if _is_informative_word(w)}
    lime_words = {_normalize_word(w) for w in lime_words if _is_informative_word(w)}
    attention_words = {
        _normalize_word(w) for w in attention_words if _is_informative_word(w)
    }

    all_words = shap_words | lime_words | attention_words
    if not all_words:
        return {"score": 0.0, "consensus_words": []}

    consensus = []
    for word in all_words:
        count = int(word in shap_words) + int(word in lime_words) + int(word in attention_words)
        if count >= 2:
            consensus.append((word, count))
    consensus.sort(key=lambda x: (-x[1], x[0]))
    consensus_words = [w for w, _ in consensus[:10]]

    # Agreement score: ratio of words agreed by at least 2 methods.
    agreement_score = round(len(consensus_words) / max(1, len(all_words)), 4)
    return {
        "score": agreement_score,
        "consensus_words": consensus_words,
        "is_low_agreement": agreement_score < 0.2,
    }


def get_emotion_signals(text: str) -> dict:
    """
    Lightweight emotion-aware stress signals using lexical cues.
    This approximates emotion-infused interpretation without requiring
    another large model at inference time.
    """
    words = [_normalize_word(w) for w in text.split()]
    words = [w for w in words if w]
    if not words:
        return {"dominant_emotion": "none", "scores": {}}

    counts = {}
    for emotion, lex in _STRESS_EMOTION_LEXICON.items():
        counts[emotion] = sum(1 for w in words if w in lex)

    total_hits = sum(counts.values())
    if total_hits == 0:
        return {"dominant_emotion": "none", "scores": {k: 0.0 for k in counts}}

    normalized = {k: round(v / total_hits, 4) for k, v in counts.items()}
    dominant = max(normalized.items(), key=lambda x: x[1])[0]

    return {"dominant_emotion": dominant, "scores": normalized}


def build_rationale(label: str, probability: float, agreement: dict, emotion_signals: dict) -> str:
    """
    Build a concise human-readable rationale from consensus explanation words.
    """
    words = agreement.get("consensus_words", [])[:4]
    dominant_emotion = emotion_signals.get("dominant_emotion", "none")
    pct = round(float(probability) * 100)

    if label == "Uncertain":
        if words:
            return (
                f"The prediction is uncertain ({pct}%) because the strongest explanation "
                f"signals are mixed. Key words include: {', '.join(words)}."
            )
        return (
            f"The prediction is uncertain ({pct}%) because explanation methods do not "
            "show a consistent stress pattern."
        )

    if words:
        return (
            f"The model predicts {label.lower()} ({pct}%) based on words such as "
            f"{', '.join(words)}, with a dominant {dominant_emotion} signal."
        )
    return (
        f"The model predicts {label.lower()} ({pct}%), but explanation overlap is low, "
        "so treat this result with caution."
    )


# ── PUBLIC API ──────────────────────────────────────────────────────────────

def explain(text, model, tokenizer, device, prediction_context=None):
    """Run all three explanation methods, returning partial results on failure."""
    explanations = {}

    try:
        explanations["shap"] = get_shap_explanation(text, model, tokenizer, device)
    except Exception as e:
        print(f"SHAP explanation failed: {e}")
        explanations["shap"] = {}

    try:
        explanations["lime"] = get_lime_explanation(text, model, tokenizer, device)
    except Exception as e:
        print(f"LIME explanation failed: {e}")
        explanations["lime"] = {}

    try:
        explanations["attention"] = get_attention_explanation(
            text, model, tokenizer, device
        )
    except Exception as e:
        print(f"Attention explanation failed: {e}")
        explanations["attention"] = {}

    explanations["agreement"] = get_explanation_agreement(explanations)
    explanations["emotion_signals"] = get_emotion_signals(text)

    if prediction_context:
        explanations["rationale"] = build_rationale(
            label=prediction_context.get("label", "Uncertain"),
            probability=prediction_context.get("probability", 0.5),
            agreement=explanations["agreement"],
            emotion_signals=explanations["emotion_signals"],
        )
    else:
        explanations["rationale"] = ""

    return explanations
