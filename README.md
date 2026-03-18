# 🧠 Real-Time Social Media Mood Analyzer

A full NLP mini-project that reads text from social media pages, classifies its
mood/sentiment (positive / negative / neutral) using machine learning, and
displays the result via a Chrome extension.

---

## 📁 Project Structure

```
mood_analyzer/
│
├── dataset/
│   └── reviews.csv              ← 80-row labelled dataset
│
├── nlp_model/
│   ├── train_model.py           ← Training script (3 models)
│   ├── saved_model/             ← Created after training
│   │   ├── best_model.pkl
│   │   ├── vectorizer.pkl
│   │   └── labels.pkl
│   └── plots/                   ← Created after training
│       ├── accuracy_comparison.png
│       ├── confusion_matrices.png
│       └── precision_recall.png
│
├── backend/
│   └── app.py                   ← Flask REST API
│
├── extension/
│   ├── manifest.json
│   ├── content.js
│   ├── popup.html
│   ├── popup.js
│   ├── style.css
│   └── icons/
│       ├── icon16.png
│       ├── icon48.png
│       └── icon128.png
│
├── report/
│   └── report.md                ← Full university report content
│
└── requirements.txt
```

---

## ⚡ Quick Start (5 steps)

### Step 1 — Install Python dependencies

```bash
# From the project root folder
pip install -r requirements.txt
```

### Step 2 — Train the NLP model

```bash
cd nlp_model
python train_model.py
```

Expected output:
```
[1/7] Loading dataset …      Loaded 82 rows  |  Labels: {'positive': 28, 'negative': 28, 'neutral': 26}
[2/7] Cleaning text …
[3/7] Extracting TF-IDF features …
[4/7] Training models …

  ── Naive Bayes ────────────────────────
     Accuracy : 0.8571
     ...
  ── Logistic Regression ────────────────
     ...
  ── SVM ────────────────────────────────
     ...

[5/7] Best model → SVM  (F1=0.88)
      Saved to nlp_model/saved_model/
[6/7] Generating plots …
      Plots saved to nlp_model/plots/
[7/7] ✅  Training complete!
```

Three PNG charts are saved to `nlp_model/plots/`.

### Step 3 — Start the Flask API

```bash
cd backend
python app.py
```

You should see:
```
Loading model … ✅
 * Running on http://0.0.0.0:5000
```

**Test it from a terminal:**
```bash
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "I absolutely love this update!"}'
```

Expected response:
```json
{
  "sentiment": "positive",
  "confidence": 0.9143,
  "emoji": "😊",
  "scores": {
    "negative": 0.0234,
    "neutral":  0.0623,
    "positive": 0.9143
  }
}
```

### Step 4 — Load the Chrome Extension

1. Open Chrome and go to `chrome://extensions/`
2. Enable **Developer mode** (toggle top-right)
3. Click **Load unpacked**
4. Select the `extension/` folder
5. The 🧠 icon should appear in your Chrome toolbar

### Step 5 — Test on websites

1. Visit Reddit, Twitter/X, WhatsApp Web, or any webpage
2. Click the 🧠 extension icon
3. Click **🔍 Analyze Page** — the extension extracts text and shows the mood
4. Or type/paste your own text in the **✏️ Custom** tab
5. Click **🏷️ Badge Comments** to annotate every comment on the page inline

---

## 🛠️ API Reference

| Method | Endpoint   | Body                     | Response                                         |
|--------|------------|--------------------------|--------------------------------------------------|
| GET    | /health    | —                        | `{"status":"ok","model_loaded":true}`            |
| POST   | /predict   | `{"text":"your text"}`   | `{"sentiment":"…","confidence":0.x,"scores":{…}}`|

---

## 🔄 Using Your Own Dataset

1. Replace `dataset/reviews.csv` with your own CSV file.
2. Ensure it has two columns: `text` and `label`.
3. Labels must be **exactly**: `positive`, `negative`, `neutral`.
4. Run `python nlp_model/train_model.py` again.

---

## 📊 Evaluation Metrics Explained

| Metric    | What it measures                                   |
|-----------|----------------------------------------------------|
| Accuracy  | % of all predictions that were correct             |
| Precision | Of predicted positives, how many were truly positive |
| Recall    | Of actual positives, how many were found            |
| F1-score  | Harmonic mean of Precision and Recall              |
| MCC       | Balanced quality measure (-1 worst, +1 perfect)    |

---

## 💡 Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| Extension shows "API Offline" | Make sure `python backend/app.py` is running |
| `saved_model/` not found | Run `python nlp_model/train_model.py` first |
| Extension can't read page | Some pages block content scripts; try another site |
| Low accuracy | Add more training data to `dataset/reviews.csv` |
