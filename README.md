# 📰 Fake News Detection System

A machine learning system that classifies news articles as **Real or Fake** with 99.64% accuracy.  
Built with a TF-IDF + SGD Classifier pipeline, exposed via both a Streamlit UI and a FastAPI REST endpoint.

---

## Features

- **NLP preprocessing pipeline** — URL removal, lowercasing, punctuation stripping, whitespace normalization
- **TF-IDF vectorization** — unigrams + bigrams, 50,000 features, trained on title + article text combined
- **SGD Classifier** — fast linear model suitable for high-dimensional sparse text data
- **Confidence scoring** — decision boundary distance mapped to a 0–100% confidence value
- **Streamlit UI** — interactive frontend with model metrics panel and confusion matrix
- **FastAPI REST API** — `/predict` endpoint for programmatic access with JSON I/O

---

## Model Performance

| Metric | Score |
|---|---|
| Accuracy | 99.64% |
| Precision (Fake) | 99.70% |
| Recall (Fake) | 99.62% |
| Precision (Real) | 99.58% |
| Recall (Real) | 99.67% |

Trained on **44,898 articles** (23,481 fake · 21,417 real)

---

## Project Structure
```
FakeNewsProject/
├── train_model.py     # Data loading, preprocessing, training, evaluation
├── app.py             # Streamlit UI
├── api.py             # FastAPI REST API
├── model.pkl          # Trained SGD Classifier
├── vectorizer.pkl     # Fitted TF-IDF Vectorizer
├── metrics.json       # Saved evaluation metrics
├── Fake.csv           # Fake news dataset
├── True.csv           # Real news dataset
└── requirements.txt
```

---

## Setup
```bash
pip install scikit-learn joblib streamlit fastapi uvicorn pandas
```

**Train the model:**
```bash
python train_model.py
```

**Run the Streamlit UI:**
```bash
streamlit run app.py
```

**Run the FastAPI server:**
```bash
uvicorn api:app --reload
```

API docs available at: `http://localhost:8000/docs`

---

## API Usage

**POST** `/predict`
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Scientists discover new treatment for common disease"}'
```

**Response:**
```json
{
  "label": "REAL",
  "confidence": 87.3,
  "is_fake": false
}
```

**GET** `/metrics` — Returns model evaluation metrics

---

## Dataset

[Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) via Kaggle.