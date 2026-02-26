import streamlit as st
import joblib
import json
import re

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)


# ── Load artifacts ────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    model      = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

@st.cache_data
def load_metrics():
    with open("metrics.json") as f:
        return json.load(f)

model, vectorizer = load_model()
metrics = load_metrics()


# ── Preprocessing (must match train_model.py) ─────────────────────────────────

def preprocess(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── UI ────────────────────────────────────────────────────────────────────────

st.title("📰 Fake News Detection System")
st.caption("Paste a news headline or article. The model will classify it as Real or Fake.")

st.divider()

user_input = st.text_area(
    "News Content",
    placeholder="Paste headline or article text here...",
    height=180
)

col1, col2 = st.columns([1, 4])
with col1:
    check = st.button("Analyze", type="primary", use_container_width=True)
with col2:
    clear = st.button("Clear", use_container_width=True)

if clear:
    st.rerun()

if check:
    if len(user_input.strip()) < 10:
        st.warning("Please enter at least a sentence for reliable detection.")
    else:
        cleaned = preprocess(user_input)
        vec     = vectorizer.transform([cleaned])

        prediction  = model.predict(vec)[0]
        # Decision function gives distance from boundary → proxy for confidence
        raw_score   = model.decision_function(vec)[0]
        confidence  = min(round(abs(raw_score) / (abs(raw_score) + 1) * 100 + 50, 1), 99.9)

        st.divider()

        if prediction == 1:
            st.success("✅ This news appears to be **REAL**")
        else:
            st.error("🚫 This news appears to be **FAKE**")

        st.markdown(f"**Confidence:** {confidence}%")
        st.progress(int(confidence))

        with st.expander("What does this mean?"):
            st.write(
                "The model analyzes word patterns and statistical features "
                "learned from ~45,000 labeled news articles. A higher confidence "
                "means the text is further from the decision boundary."
            )


# ── Model metrics sidebar ─────────────────────────────────────────────────────

with st.sidebar:
    st.header("Model Performance")
    st.metric("Accuracy",        f"{metrics['accuracy']}%")
    st.metric("Precision (Fake)", f"{metrics['precision_fake']}%")
    st.metric("Recall (Fake)",    f"{metrics['recall_fake']}%")
    st.metric("Precision (Real)", f"{metrics['precision_real']}%")
    st.metric("Recall (Real)",    f"{metrics['recall_real']}%")

    st.divider()
    st.caption("Trained on 44,898 articles · TF-IDF + SGD Classifier · Bigrams")

    cm = metrics["confusion_matrix"]
    st.markdown("**Confusion Matrix**")
    st.markdown(f"""
    |  | Predicted Fake | Predicted Real |
    |--|--|--|
    | **Actual Fake** | {cm[0][0]} | {cm[0][1]} |
    | **Actual Real** | {cm[1][0]} | {cm[1][1]} |
    """)