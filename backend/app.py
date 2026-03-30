"""
FastAPI backend for Explainable Stress Detection.

Run:
    cd backend
    uvicorn app:app --reload --port 8000
"""

import json
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model.guardrails import apply_conservative_postcheck
from model.model_loader import load_model, get_device
from model.predict import predict
from model.explain import explain
from utils.preprocessing import clean_text

app = FastAPI(
    title="Explainable Stress Detection API",
    description="Predict stress from English text with SHAP, LIME, and Attention explanations.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None
_tokenizer = None
_device = None
_decision_threshold = 0.5


@app.on_event("startup")
def startup_load_model():
    global _model, _tokenizer, _device, _decision_threshold
    _device = get_device()
    _model, _tokenizer = load_model()
    threshold_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "model",
        "saved",
        "threshold_analysis.json",
    )
    try:
        with open(threshold_path, "r", encoding="utf-8") as f:
            threshold_rows = json.load(f)
        if isinstance(threshold_rows, list) and threshold_rows:
            best_row = max(threshold_rows, key=lambda row: float(row.get("f1", 0.0)))
            _decision_threshold = float(best_row.get("threshold", 0.5))
    except Exception:
        _decision_threshold = 0.5
    print(f"Model loaded on {_device}")


class TextInput(BaseModel):
    text: str


@app.post("/predict")
def predict_stress(body: TextInput):
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="Text must not be empty.")

    cleaned = clean_text(body.text)
    prediction = predict(
        cleaned,
        _model,
        _tokenizer,
        _device,
        decision_threshold=_decision_threshold,
    )
    explanations = explain(cleaned, _model, _tokenizer, _device, prediction_context=prediction)
    prediction = apply_conservative_postcheck(cleaned, prediction, explanations)
    explanations = explain(cleaned, _model, _tokenizer, _device, prediction_context=prediction)

    return {
        "label": prediction["label"],
        "probability": prediction["probability"],
        "is_uncertain": prediction["is_uncertain"],
        "recommended_action": prediction["recommended_action"],
        "emotion_diagnostics": prediction.get("emotion_diagnostics"),
        "explanations": explanations,
    }


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": _model is not None}
