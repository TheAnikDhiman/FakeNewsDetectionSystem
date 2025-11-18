import streamlit as st
import joblib

# Load trained model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# UI Title
st.title("📰 Fake News Detection System")
st.write("Enter a news headline or paragraph below to check if it's Real or Fake.")

# Input text box
user_input = st.text_area("Paste News Content Here")

# Predict button
if st.button("Check"):
    if len(user_input) > 5:
        # Convert input to vector
        transformed_input = vectorizer.transform([user_input])
        prediction = model.predict(transformed_input)[0]

        # Show result
        if prediction == 1:
            st.success("✅ This news is likely **REAL**.")
        else:
            st.error("🚫 This news is likely **FAKE**.")
    else:
        st.warning("Please enter a longer news text for better detection.") 
