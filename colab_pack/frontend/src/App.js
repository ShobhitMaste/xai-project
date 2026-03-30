import React, { useState } from "react";
import axios from "axios";
import InputBox from "./components/InputBox";
import ResultDisplay from "./components/ResultDisplay";
import SHAPChart from "./components/SHAPChart";
import LIMEExplanation from "./components/LIMEExplanation";
import AttentionHeatmap from "./components/AttentionHeatmap";
import "./App.css";

const API_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

function App() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleAnalyze = async (text) => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await axios.post(`${API_URL}/predict`, { text });
      setResult(response.data);
    } catch (err) {
      setError(
        err.response?.data?.detail ||
          "Could not reach the server. Is the backend running?"
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>Explainable Stress Detection</h1>
        <p>
          Enter English text and get a stress prediction with SHAP, LIME, and
          Attention explanations.
        </p>
      </header>

      <InputBox onAnalyze={handleAnalyze} loading={loading} />

      {loading && (
        <div className="loading-spinner">
          <div className="spinner" />
          <span>Analyzing text &amp; generating explanations...</span>
        </div>
      )}

      {error && <div className="error-message">{error}</div>}

      {result && (
        <>
          {/*
            Normalize optional API sections so partial backend explanations
            never cause UI runtime errors.
          */}
          {(() => {
            const safeExplanations = result.explanations || {};
            return (
              <>
          <ResultDisplay
            label={result.label}
            probability={result.probability}
            isUncertain={result.is_uncertain}
            recommendedAction={result.recommended_action}
            agreement={safeExplanations.agreement}
            emotionSignals={safeExplanations.emotion_signals}
            emotionDiagnostics={result.emotion_diagnostics}
            rationale={safeExplanations.rationale}
          />

          <div className="explanations-grid">
            {safeExplanations.shap &&
              Object.keys(safeExplanations.shap).length > 0 && (
                <SHAPChart data={safeExplanations.shap} />
              )}

            {safeExplanations.lime &&
              Object.keys(safeExplanations.lime).length > 0 && (
                <LIMEExplanation data={safeExplanations.lime} />
              )}

            {safeExplanations.attention &&
              Object.keys(safeExplanations.attention).length > 0 && (
                <AttentionHeatmap data={safeExplanations.attention} />
              )}
          </div>
              </>
            );
          })()}
        </>
      )}
    </div>
  );
}

export default App;
