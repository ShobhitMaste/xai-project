import React, { useState } from "react";

const EXAMPLES = [
  "I am feeling overwhelmed with work and deadlines are piling up",
  "Had a wonderful relaxing day at the beach with friends",
  "I can't sleep because of anxiety about my upcoming exams",
  "Just finished a great workout and feeling energized",
];

function InputBox({ onAnalyze, loading }) {
  const [text, setText] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    if (text.trim()) onAnalyze(text.trim());
  };

  return (
    <div className="card">
      <h2>Enter Text</h2>
      <form onSubmit={handleSubmit} className="input-area">
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Type or paste English text here..."
          disabled={loading}
        />
        <button type="submit" className="btn-analyze" disabled={loading || !text.trim()}>
          {loading ? "Analyzing..." : "Analyze Stress"}
        </button>
      </form>

      <div style={{ marginTop: 16 }}>
        <span style={{ fontSize: "0.8rem", color: "#64748b" }}>
          Try an example:
        </span>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginTop: 8 }}>
          {EXAMPLES.map((ex, i) => (
            <button
              key={i}
              onClick={() => setText(ex)}
              disabled={loading}
              style={{
                padding: "6px 12px",
                fontSize: "0.78rem",
                borderRadius: 8,
                border: "1px solid rgba(255,255,255,0.12)",
                background: "rgba(255,255,255,0.06)",
                color: "#cbd5e1",
                cursor: "pointer",
              }}
            >
              {ex.length > 50 ? ex.slice(0, 50) + "..." : ex}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

export default InputBox;
