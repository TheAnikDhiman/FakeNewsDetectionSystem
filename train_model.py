import pandas as pd
import re
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
import joblib


# ── 1. Preprocessing ──────────────────────────────────────────────────────────

def preprocess(text: str) -> str:
    """Clean and normalize raw news text."""
    text = str(text).lower()                        # lowercase
    text = re.sub(r"http\S+|www\S+", "", text)      # remove URLs
    text = re.sub(r"[^a-z\s]", "", text)            # keep only letters
    text = re.sub(r"\s+", " ", text).strip()        # collapse whitespace
    return text


# ── 2. Load & prepare data ────────────────────────────────────────────────────

fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

fake["label"] = 0
true["label"] = 1

data = pd.concat([fake, true], axis=0).reset_index(drop=True)
data = data.sample(frac=1, random_state=42)

# Combine title + text for richer features
data["content"] = (data["title"].fillna("") + " " + data["text"].fillna(""))
data["content"] = data["content"].apply(preprocess)

print(f"Dataset size: {len(data)} articles")
print(f"Fake: {(data['label']==0).sum()} | Real: {(data['label']==1).sum()}")


# ── 3. Train / test split ─────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    data["content"],
    data["label"],
    test_size=0.2,
    random_state=42
)


# ── 4. TF-IDF vectorization ───────────────────────────────────────────────────

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.7,
    ngram_range=(1, 2),     # unigrams + bigrams
    max_features=50000
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)


# ── 5. Train model ────────────────────────────────────────────────────────────

model = SGDClassifier(loss='hinge', penalty=None, learning_rate='pa1', eta0=1.0, max_iter=100)
model.fit(X_train_vec, y_train)


# ── 6. Evaluate ───────────────────────────────────────────────────────────────

preds = model.predict(X_test_vec)
acc   = accuracy_score(y_test, preds)
report = classification_report(y_test, preds, target_names=["Fake", "Real"], output_dict=True)
cm    = confusion_matrix(y_test, preds).tolist()

print(f"\nAccuracy: {round(acc * 100, 2)}%")
print("\nClassification Report:")
print(classification_report(y_test, preds, target_names=["Fake", "Real"]))
print("Confusion Matrix:")
print(confusion_matrix(y_test, preds))


# ── 7. Save artifacts ─────────────────────────────────────────────────────────

joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

metrics = {
    "accuracy": round(acc * 100, 2),
    "precision_fake":  round(report["Fake"]["precision"] * 100, 2),
    "recall_fake":     round(report["Fake"]["recall"] * 100, 2),
    "precision_real":  round(report["Real"]["precision"] * 100, 2),
    "recall_real":     round(report["Real"]["recall"] * 100, 2),
    "confusion_matrix": cm
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\nSaved: model.pkl, vectorizer.pkl, metrics.json")