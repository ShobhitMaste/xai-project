import torch
import torch.nn.functional as F
from model.emotion_aux import predict_emotion_aux


def predict(
    text: str,
    model,
    tokenizer,
    device,
    uncertainty_margin: float = 0.08,
    decision_threshold: float = 0.5,
) -> dict:
    """
    Run stress prediction on a single text.

    Returns:
        {
          "label": "Stressed" | "Not Stressed" | "Uncertain",
          "probability": float,
          "is_uncertain": bool,
          "recommended_action": str
        }
    """
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=128
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=-1)
    probability = probs[0][1].item()  # P(Stressed)
    if abs(probability - decision_threshold) <= uncertainty_margin:
        label = "Uncertain"
        is_uncertain = True
        recommended_action = (
            "Prediction is near the decision boundary. Provide more context "
            "or longer text for a more reliable estimate."
        )
    else:
        label = "Stressed" if probability >= decision_threshold else "Not Stressed"
        is_uncertain = False
        recommended_action = "Prediction confidence is sufficient for this sample."

    output = {
        "label": label,
        "probability": round(probability, 4),
        "is_uncertain": is_uncertain,
        "recommended_action": recommended_action,
    }
    if getattr(model, "emotion_aux", None) is not None:
        output["emotion_diagnostics"] = predict_emotion_aux(text, model.emotion_aux)
    return output
