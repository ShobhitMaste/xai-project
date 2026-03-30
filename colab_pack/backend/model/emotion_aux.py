import os
from typing import Dict, List

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


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

    counts = {k: 0 for k in _EMOTION_LEXICON.keys()}
    for w in words:
        for emotion, lex in _EMOTION_LEXICON.items():
            if w in lex:
                counts[emotion] += 1
    if sum(counts.values()) == 0:
        return "none"
    return max(counts.items(), key=lambda x: x[1])[0]


def train_emotion_aux(texts: List[str]) -> Dict[str, object]:
    labels = [_derive_emotion_label(t) for t in texts]
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    x = vectorizer.fit_transform(texts)
    clf = LogisticRegression(max_iter=500)
    clf.fit(x, labels)
    return {"vectorizer": vectorizer, "classifier": clf, "labels": EMOTION_LABELS}


def save_emotion_aux(bundle: Dict[str, object], save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "emotion_aux.joblib")
    joblib.dump(bundle, path)


def load_emotion_aux(save_dir: str):
    path = os.path.join(save_dir, "emotion_aux.joblib")
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def predict_emotion_aux(text: str, bundle: Dict[str, object]) -> Dict[str, object]:
    vectorizer = bundle["vectorizer"]
    clf = bundle["classifier"]
    x = vectorizer.transform([text])
    probs = clf.predict_proba(x)[0]
    classes = clf.classes_
    prob_map = {str(c): float(p) for c, p in zip(classes, probs)}
    dominant = max(prob_map.items(), key=lambda x: x[1])[0]
    return {
        "dominant_emotion_model": dominant,
        "probabilities": {k: round(v, 4) for k, v in prob_map.items()},
    }
