📰 Fake News Detection (Machine Learning Project)

This project identifies whether a news article is Real or Fake using NLP techniques and traditional Machine Learning. The goal was to build a simple, interpretable, and efficient model suitable for academic submissions and real-world learning.

🚀 Features

Text preprocessing (cleaning, stopwords removal, stemming)

TF-IDF vector representation

Logistic Regression classifier

Model evaluation with accuracy, confusion matrix, and classification report

Custom input prediction

Organized Jupyter Notebook with clear steps

🧠 Tech Stack

Python

Scikit-learn

Pandas & NumPy

NLTK

Jupyter Notebook

📂 Project Structure
fake-news-detection/
│
├── notebook.ipynb          # Data cleaning, EDA, training & evaluation
├── app.py (optional)       # Script for custom input prediction
├── requirements.txt        # Dependencies
├── README.md               # Project documentation
└── dataset-link.txt        # (Add dataset link instead of raw data)

📊 Model

The model uses TF-IDF vectorization + Logistic Regression, chosen because:

It performs well on text classification

It's fast to train

It’s easy to interpret for academic submissions

(If you share your model accuracy, I’ll add it here.)

🧪 How to Run Locally
1. Install dependencies
pip install -r requirements.txt

2. Run the notebook

Open notebook.ipynb and run all cells.

python app.py

📁 Dataset

Dataset used: Fake News Classification dataset
You can download it from Kaggle

Add link here: [<insert-dataset-link>](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

🎯 Results

Preprocessing → TF-IDF vectors

Model → Logistic Regression

Outcome → Fake/Real classification

Test Accuracy: **99.35%**
👨‍💻 Author

Anik Dhiman
