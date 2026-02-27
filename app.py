from flask import Flask, render_template, request
import spacy
import torch
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import nltk
from nltk.tokenize import sent_tokenize
import os

# -------------------------
# INITIAL SETUP
# -------------------------

app = Flask(__name__)

# Download NLTK data (first time only)
nltk.download("punkt")

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# -------------------------
# LOAD TRAINED RoBERTa MODEL
# -------------------------

MODEL_NAME = "hamzab/roberta-fake-news-classification"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

model.eval()

print("Model Loaded:", model.config._name_or_path)

# -------------------------
# GOOGLE FACT CHECK API KEY
# -------------------------

API_KEY = os.environ.get("FACT_CHECK_API_KEY")

if not API_KEY:
    print("⚠ WARNING: Google Fact Check API key not set!")

# -------------------------
# CLASSIFICATION FUNCTION
# -------------------------

def classify_text(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=1)

    confidence = torch.max(probs).item()
    label_id = torch.argmax(probs).item()

    label = model.config.id2label[label_id]

    return label, round(confidence * 100, 2)

# -------------------------
# CLAIM EXTRACTION
# -------------------------

def extract_claims(text):
    sentences = sent_tokenize(text)
    claims = []

    for sentence in sentences:
        if len(sentence.split()) > 6:  # slightly stricter
            claims.append(sentence)

    return claims

# -------------------------
# GOOGLE FACT CHECK API
# -------------------------

def fact_check_api(claim):
    if not API_KEY:
        return []

    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

    # 🔥 Shorten query for better match
    short_claim = claim[:120]

    params = {
        "query": short_claim,
        "key": API_KEY,
        "languageCode": "en",
        "pageSize": 5
    }

    try:
        response = requests.get(url, params=params)

        if response.status_code != 200:
            print("API Status Error:", response.status_code)
            print(response.text)
            return []

        data = response.json()
        print("Fact Check API Response:", data)

        results = []

        if "claims" in data and len(data["claims"]) > 0:
            for item in data["claims"]:
                for review in item.get("claimReview", []):
                    results.append({
                        "publisher": review.get("publisher", {}).get("name", "Unknown"),
                        "rating": review.get("textualRating", "No rating"),
                        "url": review.get("url", "#")
                    })

        return results

    except Exception as e:
        print("API Exception:", e)
        return []

# -------------------------
# ROUTES
# -------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    claims_data = []
    entities = []
    original_text = None

    if request.method == "POST":
        text = request.form.get("news_text")
        original_text = text

        if text and text.strip():

            # 1️⃣ RoBERTa Classification
            prediction, confidence = classify_text(text)

            # 2️⃣ Named Entity Recognition
            doc = nlp(text)
            entities = [(ent.text, ent.label_) for ent in doc.ents]

            # 3️⃣ Claim Extraction
            claims = extract_claims(text)

            # 4️⃣ Google Fact Check Verification
            for claim in claims:
                verification = fact_check_api(claim)

                claims_data.append({
                    "claim": claim,
                    "verification": verification
                })

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        claims_data=claims_data,
        entities=entities,
        original_text=original_text
    )

# -------------------------

if __name__ == "__main__":
    app.run(debug=True)