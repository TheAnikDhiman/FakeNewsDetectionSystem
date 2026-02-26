from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import re
import json

app = FastAPI(
    title="Fake News Detection API",
    description="REST API for classifying news articles as Real or Fake using TF-IDF + SGD Classifier",
    version="1.0.0"
)

# ── Load artifacts ────────────────────────────────────────────────────────────

model      = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

with open("metrics.json") as f:
    metrics = json.load(f)


# ── Preprocessing (must match train_model.py) ─────────────────────────────────

def preprocess(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Schemas ───────────────────────────────────────────────────────────────────

class NewsInput(BaseModel):
    text: str

class PredictionOutput(BaseModel):
    label: str          # "REAL" or "FAKE"
    confidence: float   # 0–100
    is_fake: bool


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "message": "Fake News Detection API is running"}


@app.post("/predict", response_model=PredictionOutput)
def predict(payload: NewsInput):
    if len(payload.text.strip()) < 10:
        raise HTTPException(status_code=422, detail="Text too short. Provide at least a sentence.")

    cleaned    = preprocess(payload.text)
    vec        = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]
    raw_score  = model.decision_function(vec)[0]
    confidence = min(round(abs(raw_score) / (abs(raw_score) + 1) * 100 + 50, 1), 99.9)

    return PredictionOutput(
        label      = "REAL" if prediction == 1 else "FAKE",
        confidence = confidence,
        is_fake    = prediction == 0
    )


@app.get("/metrics")
def get_metrics():
    return metrics