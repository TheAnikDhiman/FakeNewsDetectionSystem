import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# Add labels
fake["label"] = 0
true["label"] = 1

# Combine the data
data = pd.concat([fake, true], axis=0)
data = data.sample(frac=1, random_state=42)  # Shuffle

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    data["text"],
    data["label"],
    test_size=0.2,
    random_state=42
)

# Convert text to numeric features
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train the model
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train, y_train)

# Test accuracy
pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)
print("Model Accuracy:", round(acc * 100, 2), "%")

# Save model and vectorizer
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model and vectorizer saved successfully!")