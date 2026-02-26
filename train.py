import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
data = pd.read_csv("data/news.csv")

# Convert labels to numeric
data['label'] = data['label'].map({
    'FAKE': 0,
    'REAL': 1
})

# Features and labels
x = data['text']
y = data['label']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0
)

# TF-IDF Vectorization
vectorization = TfidfVectorizer(stop_words="english")

x_train = vectorization.fit_transform(x_train)
x_test = vectorization.transform(x_test)   # ✅ correct

# Model
model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000
)

model.fit(x_train, y_train)

# Prediction
pred = model.predict(x_test)

print("Accuracy:", accuracy_score(y_test, pred))

# Save model and vectorizer together
joblib.dump((model, vectorization), "model/model.pkl")

print("Model Saved Successfully")
