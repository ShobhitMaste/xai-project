"""Inference-time guardrails shared by the API and evaluation scripts."""

import re


_NEGATED_STRESS_PATTERNS = (
    r"\bnot\s+stressed\b",
    r"\bno\s+stress\b",
    r"\bstress\s*free\b",
    r"\bnever\s+stressed\b",
)


def has_explicit_not_stressed_claim(text: str) -> bool:
    t = text.lower().strip()
    if not t:
        return False
    if any(marker in t for marker in (" but ", " however ", " though ", " although ")):
        return False
    return any(re.search(pattern, t) for pattern in _NEGATED_STRESS_PATTERNS)


def apply_conservative_postcheck(text: str, prediction: dict, explanations: dict) -> dict:
    """
    Adjust borderline "Stressed" predictions using negation cues and
    neutral auxiliary signals.
    """
    if prediction.get("label") != "Stressed":
        return prediction

    probability = float(prediction.get("probability", 0.0))

    if has_explicit_not_stressed_claim(text):
        updated = dict(prediction)
        updated["probability"] = round(1.0 - probability, 4)
        updated["label"] = "Not Stressed"
        updated["is_uncertain"] = False
        updated["recommended_action"] = (
            "Detected an explicit negated-stress statement (e.g., 'not stressed'), "
            "so the final decision is adjusted toward Not Stressed."
        )
        return updated

    if probability >= 0.75:
        return prediction

    emotion_signal = explanations.get("emotion_signals", {}).get("dominant_emotion", "none")
    aux_emotion = (prediction.get("emotion_diagnostics") or {}).get(
        "dominant_emotion_model", "none"
    )
    if not ((emotion_signal == "none") and (aux_emotion == "none")):
        return prediction

    updated = dict(prediction)
    updated["label"] = "Uncertain"
    updated["is_uncertain"] = True
    updated["recommended_action"] = (
        "Prediction is borderline stressed, but lexical and auxiliary emotion "
        "signals are neutral. Provide a longer sample for a more reliable result."
    )
    return updated
