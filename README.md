# Explainable Stress Detection from English Text

A full-stack system that predicts psychological stress from English text and explains its predictions using **SHAP**, **LIME**, and **Attention visualization**.

Built with BERT (fine-tuned), FastAPI, and React.

---

## Project Structure

```
project/
├── backend/
│   ├── app.py                  # FastAPI server
│   ├── requirements.txt
│   ├── model/
│   │   ├── train.py            # Fine-tune BERT
│   │   ├── predict.py          # Prediction pipeline
│   │   ├── explain.py          # SHAP + LIME + Attention
│   │   ├── model_loader.py     # Load saved model
│   │   └── saved/              # (created after training)
│   └── utils/
│       └── preprocessing.py    # Text cleaning
│
├── frontend/
│   ├── package.json
│   ├── public/index.html
│   └── src/
│       ├── App.js              # Main React app
│       ├── App.css
│       └── components/
│           ├── InputBox.js
│           ├── ResultDisplay.js
│           ├── SHAPChart.js
│           ├── LIMEExplanation.js
│           └── AttentionHeatmap.js
│
└── README.md
```

---

## Prerequisites

- **Python 3.9+**
- **Node.js 16+** and npm
- (Optional) NVIDIA GPU with CUDA for faster training

---

## Setup & Run

### Step 1: Install Backend Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### Step 2: Train the Model (Real Datasets)

```bash
cd backend
python model/train.py
```

Default behavior now trains on the **real Dreaddit dataset** from HuggingFace, and saves the best model to `backend/model/saved/`.

You can merge multiple datasets in one run:

```bash
python model/train.py \
  --csv_paths data/stress_a.csv data/stress_b.csv \
  --extra_hf_datasets your-org/stress-dataset another-org/stress-v2
```

Enable auxiliary stress+emotion multi-task diagnostics:

```bash
python model/train.py --multi_task --csv_paths ../resources/dreaddit-train.csv
```

Notes:
- Local CSV files must have `text` and `label` columns by default.
- If your column names differ, pass `--text_col` and `--label_col`.
- Labels can be `0/1`, `True/False`, `yes/no`, `stressed/not stressed`, `stress_positive/stress_negative`.
- Rows with unknown label formats are skipped with a warning during loading.
- The script uses stratified train/val/test split and reports Accuracy, F1, Precision, Recall, and ROC-AUC.
- Training includes class-weighted loss (for imbalance) and early stopping by validation F1.
- A threshold report (`threshold_analysis.json`) is saved so you can tune the stress cutoff beyond 0.5.
- You can tune early stopping with:

```bash
python model/train.py --early_stopping_patience 3
```

Inspect datasets before training:

```bash
cd backend
python model/dataset_inspect.py --csv_paths ../resources/dreaddit-train.csv ../resources/dreaddit-test.csv
```

### Step 3: Start the Backend

```bash
cd backend
uvicorn app:app --reload --port 8000
```

The API will be available at `http://localhost:8000`. Test it:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"I am feeling overwhelmed with work\"}"
```

### Step 4: Install & Start the Frontend

```bash
cd frontend
npm install
npm start
```

The React app opens at `http://localhost:3000` and talks to the backend at `http://localhost:8000`.

### Step 5: Quick Reliability Check

Backend smoke checks:

```bash
cd backend
python -m unittest tests.test_predict_smoke tests.test_api_smoke
```

One API sample check:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"I am overwhelmed with deadlines and cannot sleep.\"}"
```

---

## API Reference

### `POST /predict`

**Request:**
```json
{
  "text": "I am feeling overwhelmed with work"
}
```

**Response:**
```json
{
  "label": "Stressed",
  "probability": 0.92,
  "is_uncertain": false,
  "recommended_action": "Prediction confidence is sufficient for this sample.",
  "emotion_diagnostics": {
    "dominant_emotion_model": "anxiety",
    "probabilities": {"anxiety": 0.73, "pressure": 0.14, "burnout": 0.08, "distress": 0.03, "none": 0.02}
  },
  "explanations": {
    "shap": {"overwhelmed": 0.15, "work": 0.08, ...},
    "lime": {"overwhelmed": 0.23, "feeling": 0.05, ...},
    "attention": {"overwhelmed": 1.0, "feeling": 0.45, ...},
    "agreement": {
      "score": 0.42,
      "consensus_words": ["overwhelmed", "work"],
      "is_low_agreement": false
    },
    "emotion_signals": {
      "dominant_emotion": "anxiety",
      "scores": {"anxiety": 0.5, "burnout": 0.25, "pressure": 0.25, "distress": 0.0}
    },
    "rationale": "The model predicts stressed (92%) based on words such as overwhelmed, work..."
  }
}
```

### `GET /health`

Returns `{"status": "healthy", "model_loaded": true}`.

---

## How It Works

### Model
- **Architecture:** `bert-base-uncased` fine-tuned for binary classification (Stressed vs. Not Stressed)
- **Training:** Multi-dataset corpus merge (Dreaddit + extra CSV/HF datasets), AdamW optimizer, linear warmup schedule, gradient clipping
- **Model selection:** Best checkpoint chosen by validation F1 with early stopping, then final test metrics are reported
- **Optional multi-task mode:** `--multi_task` trains an auxiliary emotion diagnostic model and exposes inference diagnostics in API

### Explainability

| Method | Technique | What It Shows |
|--------|-----------|---------------|
| **SHAP** | Partition-based Shapley values | Per-word contribution to the Stressed class (positive = stress-inducing) |
| **LIME** | Local perturbation-based approximation | Feature weights from a local linear model around the input |
| **Attention** | BERT's last-layer [CLS] attention weights | Which words the model "looked at" most during classification |
| **Agreement** | Overlap across SHAP/LIME/Attention top words | How consistent the explanations are |
| **Emotion Signals** | Stress-emotion lexical cues (anxiety/burnout/pressure/distress) | Dominant stress-related emotional pattern in the text |
| **Rationale** | Rule-based synthesis from consensus features | Human-readable summary of why the model predicted the label |

### Interpretation Caveats
- Attention highlights are useful cues, but they are **not proof of causal influence**.
- SHAP and LIME are local approximations; explanations can vary with perturbations or text phrasing.
- Low agreement across methods indicates lower explanation reliability for that sample.

### Frontend Visualizations
- **SHAP Chart:** Horizontal bar chart (red = stress, blue = not stress)
- **LIME Chart:** Horizontal bar chart (orange = stress, teal = not stress)
- **Attention Heatmap:** Inline colored text (blue-to-red gradient by attention weight)

---

## References

This project draws on techniques from recent research in explainable stress detection:

1. Turcan et al. (NAACL 2021) – Emotion-infused multi-task BERT for stress detection
2. Winata et al. (ICASSP 2018) – Attention-based LSTM for stress detection from speech
3. Park et al. (IEEE TAC 2024) – SHAP and attention for multimodal stress interpretation
4. Wang et al. (arXiv 2024) – Cognition chain prompting for explainable stress detection
5. Turcan et al. (EMNLP 2019) – Dreaddit: Reddit dataset for stress analysis

---

## License

This project is for educational and research purposes.
