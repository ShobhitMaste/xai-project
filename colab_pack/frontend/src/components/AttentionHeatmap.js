import React from "react";

function interpolateColor(score) {
  // 0 -> cool blue (#3b82f6), 1 -> hot red (#ef4444)
  const r = Math.round(59 + (239 - 59) * score);
  const g = Math.round(130 + (68 - 130) * score);
  const b = Math.round(246 + (68 - 246) * score);
  return `rgb(${r}, ${g}, ${b})`;
}

function AttentionHeatmap({ data }) {
  const entries = Object.entries(data);

  return (
    <div className="card">
      <h2>Attention Heatmap</h2>
      <p style={{ fontSize: "0.82rem", color: "#94a3b8", marginBottom: 12 }}>
        Words colored by BERT attention weight. Red = high attention; blue = low
        attention.
      </p>
      <div className="attention-text">
        {entries.map(([word, score], idx) => (
          <span
            key={idx}
            className="attention-word"
            style={{
              background: interpolateColor(score),
              color: score > 0.5 ? "#fff" : "#f1f5f9",
              fontWeight: score > 0.6 ? 600 : 400,
            }}
            title={`Attention: ${score.toFixed(4)}`}
          >
            {word}
          </span>
        ))}
      </div>

      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
          marginTop: 20,
          fontSize: "0.78rem",
          color: "#94a3b8",
        }}
      >
        <span>Low</span>
        <div
          style={{
            flex: 1,
            height: 10,
            borderRadius: 5,
            background: "linear-gradient(90deg, #3b82f6, #ef4444)",
          }}
        />
        <span>High</span>
      </div>
    </div>
  );
}

export default AttentionHeatmap;
