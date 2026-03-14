from flask import Flask, render_template, request
import spacy
import torch
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import nltk
from nltk.tokenize import sent_tokenize
import os
import time
import re
import wikipediaapi
import urllib.parse  
# Added this to help read the secret file
from dotenv import load_dotenv 

# Load the variables from the .env file
load_dotenv()

# Ensure NLTK is ready
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

app = Flask(__name__)

# --- API CONFIGURATION ---
# Now it grabs the key safely from your environment
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
wiki = wikipediaapi.Wikipedia(user_agent="FactChecker/1.0", language='en')

# --- SYSTEM METRICS ---
articles_analyzed = 0
fake_count = 0
real_count = 0
start_time = time.time()

suspicious_patterns = {
    "miracle cure": "Claims of 'miracle cure' are often used in medical misinformation to bypass scientific rigor.",
    "eliminates all types of cancer": "Biological impossibility; cancer is a group of diseases, and no single cure exists for all.",
    "suppressed by big pharma": "Classic conspiracy narrative used to explain lack of evidence.",
    "secret government": "Vague references to 'secret' authorities are red flags.",
    "microchip": "Commonly linked to debunked vaccine-related conspiracy theories.",
    "shocking discovery": "Sensationalist language used to trigger emotional responses."
}

def detect_suspicious(text):
    highlighted = text
    explanations = []
    for phrase, explanation in suspicious_patterns.items():
        if phrase.lower() in text.lower():
            pattern = re.compile(re.escape(phrase), re.IGNORECASE)
            highlighted = pattern.sub(f'<span class="suspicious">{phrase}</span>', highlighted)
            if explanation not in explanations:
                explanations.append(explanation)
    return highlighted, explanations

def get_google_fact_check(claim):
    """Queries Google Fact Check Tools API with Wikipedia fallback"""
    if not GOOGLE_API_KEY:
        return get_wiki_verification(claim)

    try:
        # Optimization: Use first 10 words for better matching
        search_query = " ".join(claim.split()[:10])
        query = urllib.parse.quote(search_query)
        
        url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?query={query}&key={GOOGLE_API_KEY}"
        response = requests.get(url)
        data = response.json()

        if "claims" in data and len(data["claims"]) > 0:
            first_claim = data["claims"][0]
            review = first_claim["claimReview"][0]
            publisher = review["publisher"]["name"]
            rating = review["textualRating"]
            return f"Google Check: {rating} (via {publisher})"
        
        return get_wiki_verification(claim)
    except:
        return get_wiki_verification(claim)

def get_wiki_verification(claim):
    try:
        query = " ".join(claim.split()[:4])
        page = wiki.page(query)
        if page.exists():
            return f"Wikipedia: {page.title} (Verified Source)"
        return "No specific match found in archives."
    except:
        return "Verification service unavailable."

# Model Setup
try:
    nlp = spacy.load("en_core_web_sm")
except:
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

MODEL_NAME = "hamzab/roberta-fake-news-classification"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

def classify_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    confidence = torch.max(probs).item()
    idx = torch.argmax(probs).item()
    label = model.config.id2label[idx].upper() 
    label = "FAKE" if ("0" in label or "FAKE" in label) else "REAL"
    return label, round(confidence * 100, 2)

@app.route("/", methods=["GET", "POST"])
def index():
    global articles_analyzed, fake_count, real_count
    data = {"prediction": None, "confidence": None, "claims_data": [], "entities": [], "highlighted_text": None, "explanation_points": [], "original_text": None}

    if request.method == "POST":
        text = request.form.get("news_text")
        if text and text.strip():
            data["original_text"] = text
            data["prediction"], data["confidence"] = classify_text(text)
            articles_analyzed += 1
            if data["prediction"] == "FAKE": fake_count += 1
            else: real_count += 1
            data["highlighted_text"], data["explanation_points"] = detect_suspicious(text)
            
            sentences = sent_tokenize(text)
            for sent in sentences[:4]:
                data["claims_data"].append({
                    "claim": sent,
                    "source": get_google_fact_check(sent)
                })

            doc = nlp(text)
            data["entities"] = [(ent.text, ent.label_) for ent in doc.ents]

    metrics_summary = {"analyzed": articles_analyzed, "distribution": f"F:{fake_count} | R:{real_count}", "session": f"{int(time.time() - start_time)}s"}
    return render_template("index.html", **data, metrics=metrics_summary, deployment={"status": "Active (Local)", "provider": "Flask Dev Server"})

if __name__ == "__main__":
    # If running in Docker, we need 0.0.0.0
    app.run(host='0.0.0.0', port=5000, debug=True)