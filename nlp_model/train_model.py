"""
=============================================================
  Real-Time Social Media Mood Analyzer — NLP Model Trainer
  train_model.py
=============================================================
  Trains three classifiers:
    1. Multinomial Naive Bayes
    2. Logistic Regression
    3. Support Vector Machine (SVM)

  Saves the best model + vectorizer to:
    nlp_model/saved_model/best_model.pkl
    nlp_model/saved_model/vectorizer.pkl

  Outputs:
    - Accuracy/Precision/Recall/F1/MCC for each model
    - Comparison bar chart
    - Confusion matrices (one per model)
    - Precision/Recall chart
=============================================================
"""

import os
import pickle
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, matthews_corrcoef,
    classification_report
)

warnings.filterwarnings("ignore")

# ── Download required NLTK data ──────────────────────────────────────────────
for pkg in ["punkt", "stopwords", "punkt_tab"]:
    nltk.download(pkg, quiet=True)

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "..", "dataset", "reviews.csv")
SAVE_DIR    = os.path.join(BASE_DIR, "saved_model")
PLOT_DIR    = os.path.join(BASE_DIR, "plots")
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# ── 1. Load Dataset ──────────────────────────────────────────────────────────
print("\n[1/7] Loading dataset …")
df = pd.read_csv(DATASET_PATH)
df.dropna(inplace=True)
df.drop_duplicates(subset="text", inplace=True)
print(f"      Loaded {len(df)} rows  |  Labels: {df['label'].value_counts().to_dict()}")

# ── 2. Text Cleaning & Preprocessing ─────────────────────────────────────────
print("[2/7] Cleaning text …")
stemmer   = PorterStemmer()
stop_words = set(stopwords.words("english"))

def clean_text(text: str) -> str:
    """
    Pipeline:
      • lower-case
      • remove punctuation / digits
      • tokenize
      • remove stopwords
      • stem each token
    """
    import re
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)          # keep only letters
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(w) for w in tokens
              if w not in stop_words and len(w) > 2]
    return " ".join(tokens)

df["clean_text"] = df["text"].apply(clean_text)

# ── 3. TF-IDF Feature Extraction ─────────────────────────────────────────────
print("[3/7] Extracting TF-IDF features …")
vectorizer = TfidfVectorizer(
    max_features=5000,   # top 5000 terms
    ngram_range=(1, 2),  # unigrams + bigrams
    sublinear_tf=True    # apply log normalization
)
X = vectorizer.fit_transform(df["clean_text"])
y = df["label"]

# ── 4. Train / Test Split ────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print(f"      Train={X_train.shape[0]}  Test={X_test.shape[0]}")

# ── 5. Define Models ─────────────────────────────────────────────────────────
print("[4/7] Training models …\n")
models = {
    "Naive Bayes":           MultinomialNB(alpha=1.0),
    "Logistic Regression":   LogisticRegression(max_iter=1000, C=1.0, random_state=42),
    "SVM":                   LinearSVC(C=1.0, max_iter=2000, random_state=42),
}

# ── 6. Train, Evaluate, Store Results ────────────────────────────────────────
results  = {}
best_f1  = 0.0
best_name = ""
best_model = None
label_names = sorted(y.unique())

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    mcc  = matthews_corrcoef(y_test, y_pred)
    cm   = confusion_matrix(y_test, y_pred, labels=label_names)

    results[name] = dict(accuracy=acc, precision=prec, recall=rec, f1=f1, mcc=mcc, cm=cm, model=model, y_pred=y_pred)

    print(f"  ── {name} ──────────────────────────")
    print(f"     Accuracy : {acc:.4f}")
    print(f"     Precision: {prec:.4f}")
    print(f"     Recall   : {rec:.4f}")
    print(f"     F1-score : {f1:.4f}")
    print(f"     MCC      : {mcc:.4f}")
    print(classification_report(y_test, y_pred, target_names=label_names, zero_division=0))

    if f1 > best_f1:
        best_f1   = f1
        best_name = name
        best_model = model

# ── 7. Save Best Model ────────────────────────────────────────────────────────
print(f"\n[5/7] Best model → {best_name}  (F1={best_f1:.4f})")
with open(os.path.join(SAVE_DIR, "best_model.pkl"), "wb") as f:
    pickle.dump(best_model, f)
with open(os.path.join(SAVE_DIR, "vectorizer.pkl"), "wb") as f:
    pickle.dump(vectorizer, f)
# Save label list so the API can decode predictions
with open(os.path.join(SAVE_DIR, "labels.pkl"), "wb") as f:
    pickle.dump(label_names, f)
print(f"      Saved to {SAVE_DIR}/")

# ── 8. Visualisations ─────────────────────────────────────────────────────────
print("[6/7] Generating plots …")

## 8a. Accuracy Comparison Bar Chart
metric_names = ["accuracy", "precision", "recall", "f1"]
model_names  = list(results.keys())
x = np.arange(len(metric_names))
width = 0.22

fig, ax = plt.subplots(figsize=(10, 5))
for i, mname in enumerate(model_names):
    vals = [results[mname][m] for m in metric_names]
    bars = ax.bar(x + i * width, vals, width, label=mname)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7)

ax.set_xlabel("Metric")
ax.set_ylabel("Score")
ax.set_title("Model Performance Comparison")
ax.set_xticks(x + width)
ax.set_xticklabels([m.title() for m in metric_names])
ax.set_ylim(0, 1.15)
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "accuracy_comparison.png"), dpi=150)
plt.close()

## 8b. Confusion Matrices (one per model)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, res) in zip(axes, results.items()):
    sns.heatmap(res["cm"], annot=True, fmt="d", cmap="Blues",
                xticklabels=label_names, yticklabels=label_names, ax=ax)
    ax.set_title(f"Confusion Matrix\n{name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "confusion_matrices.png"), dpi=150)
plt.close()

## 8c. Precision / Recall per Model (grouped bars)
fig, ax = plt.subplots(figsize=(8, 4))
bar_w = 0.35
idxs = np.arange(len(model_names))
precs = [results[m]["precision"] for m in model_names]
recs  = [results[m]["recall"]    for m in model_names]
ax.bar(idxs - bar_w / 2, precs, bar_w, label="Precision", color="steelblue")
ax.bar(idxs + bar_w / 2, recs,  bar_w, label="Recall",    color="coral")
ax.set_xticks(idxs)
ax.set_xticklabels(model_names, rotation=10)
ax.set_ylim(0, 1.15)
ax.set_ylabel("Score")
ax.set_title("Precision vs Recall")
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "precision_recall.png"), dpi=150)
plt.close()

print(f"      Plots saved to {PLOT_DIR}/")
print("\n[7/7] ✅  Training complete!\n")
