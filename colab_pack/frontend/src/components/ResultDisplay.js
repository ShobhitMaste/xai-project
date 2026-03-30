import React from "react";

function ResultDisplay({
  label,
  probability,
  isUncertain,
  recommendedAction,
  agreement,
  emotionSignals,
  emotionDiagnostics,
  rationale,
}) {
  const isStressed = label === "Stressed";
  const isUncertainLabel = label === "Uncertain";
  const pct = Math.round(probability * 100);
  const agreementPct = Math.round((agreement?.score || 0) * 100);
  const dominantEmotion = emotionSignals?.dominant_emotion || "none";

  return (
    <div className="card">
      <h2>Prediction Result</h2>
      <div className="result-banner">
        <span
          className={`result-label ${
            isUncertainLabel ? "uncertain" : isStressed ? "stressed" : "not-stressed"
          }`}
        >
          {label}
        </span>
        <div className="prob-bar-container">
          <div className="prob-bar-bg">
            <div
              className={`prob-bar-fill ${
                isUncertainLabel ? "uncertain" : isStressed ? "stressed" : "not-stressed"
              }`}
              style={{ width: `${pct}%` }}
            />
          </div>
          <div className="prob-text">
            Stress probability: <strong>{pct}%</strong>
          </div>
          {isUncertain && <div className="prob-text">{recommendedAction}</div>}
        </div>
      </div>

      <div style={{ marginTop: 14, fontSize: "0.85rem", color: "#94a3b8" }}>
        {rationale && (
          <div style={{ marginBottom: 8, color: "#cbd5e1" }}>
            <strong>Rationale:</strong> {rationale}
          </div>
        )}
        <div>
          Explanation agreement: <strong>{agreementPct}%</strong>
        </div>
        {agreement?.is_low_agreement && (
          <div style={{ marginTop: 4, color: "#fca5a5" }}>
            Low agreement across SHAP/LIME/Attention. Interpret the prediction cautiously.
          </div>
        )}
        {agreement?.consensus_words?.length > 0 && (
          <div style={{ marginTop: 4 }}>
            Consensus words: <strong>{agreement.consensus_words.join(", ")}</strong>
          </div>
        )}
        <div style={{ marginTop: 4 }}>
          Dominant emotion signal: <strong>{dominantEmotion}</strong>
        </div>
        {emotionDiagnostics?.dominant_emotion_model && (
          <div style={{ marginTop: 4 }}>
            Auxiliary emotion model:{" "}
            <strong>{emotionDiagnostics.dominant_emotion_model}</strong>
          </div>
        )}
      </div>
    </div>
  );
}

export default ResultDisplay;
