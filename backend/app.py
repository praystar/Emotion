"""
=============================================================
  Real-Time Social Media Mood Analyzer — Flask Backend API
  backend/app.py
=============================================================
  Endpoints:
    POST /predict   → { text }  →  { sentiment, confidence, scores }
    GET  /health    → status check
=============================================================
"""

import os
import re
import pickle

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from flask import Flask, request, jsonify
from flask_cors import CORS

# ── Download NLTK data silently on first run ──────────────────────────────────
for pkg in ["punkt", "stopwords", "punkt_tab"]:
    nltk.download(pkg, quiet=True)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "..", "nlp_model", "saved_model")

# ── Load model artefacts ──────────────────────────────────────────────────────
def load_pickle(filename):
    path = os.path.join(MODEL_DIR, filename)
    with open(path, "rb") as f:
        return pickle.load(f)

print("Loading model …", end=" ", flush=True)
model      = load_pickle("best_model.pkl")
vectorizer = load_pickle("vectorizer.pkl")
labels     = load_pickle("labels.pkl")          # e.g. ['negative','neutral','positive']
print("✅")

# ── Text cleaning (must match train_model.py) ─────────────────────────────────
stemmer    = PorterStemmer()
stop_words = set(stopwords.words("english"))

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(w) for w in tokens
              if w not in stop_words and len(w) > 2]
    return " ".join(tokens)

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)          # allow requests from Chrome extension

# ── Emoji map for fun ─────────────────────────────────────────────────────────
EMOJI = {"positive": "😊", "negative": "😠", "neutral": "😐"}

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": True})

@app.route("/predict", methods=["POST"])
def predict():
    """
    Request body  : { "text": "..." }
    Response body : {
        "sentiment"  : "positive" | "negative" | "neutral",
        "confidence" : 0.0–1.0,
        "emoji"      : "😊",
        "scores"     : { "positive": 0.x, "negative": 0.x, "neutral": 0.x }
    }
    """
    data = request.get_json(silent=True)
    if not data or "text" not in data:
        return jsonify({"error": "Provide a JSON body with key 'text'."}), 400

    raw_text = str(data["text"]).strip()
    if len(raw_text) < 2:
        return jsonify({"error": "Text is too short."}), 400

    cleaned = clean_text(raw_text)
    vec     = vectorizer.transform([cleaned])

    # Predict label
    pred_label = model.predict(vec)[0]

    # Confidence: use decision_function if available (SVM), else predict_proba
    if hasattr(model, "predict_proba"):
        proba  = model.predict_proba(vec)[0]
        scores = {lbl: round(float(p), 4) for lbl, p in zip(labels, proba)}
        confidence = round(float(max(proba)), 4)
    elif hasattr(model, "decision_function"):
        df_vals  = model.decision_function(vec)[0]
        # Softmax-style normalisation so scores sum to ~1
        import numpy as np
        exp_vals = np.exp(df_vals - np.max(df_vals))
        proba    = exp_vals / exp_vals.sum()
        scores   = {lbl: round(float(p), 4) for lbl, p in zip(labels, proba)}
        confidence = round(float(max(proba)), 4)
    else:
        scores     = {lbl: 0.0 for lbl in labels}
        confidence = 1.0

    return jsonify({
        "sentiment":  pred_label,
        "confidence": confidence,
        "emoji":      EMOJI.get(pred_label, "🤔"),
        "scores":     scores
    })

# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
