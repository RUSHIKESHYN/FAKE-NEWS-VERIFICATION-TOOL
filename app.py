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

app = Flask(__name__)

# --- SYSTEM METRICS ---
articles_analyzed = 0
fake_count = 0
real_count = 0
start_time = time.time()

suspicious_patterns = {
    "miracle cure": "Claims of 'miracle cure' are often used in medical misinformation.",
    "eliminates all types of cancer": "Biological impossibility; cancer is a group of diseases.",
    "suppressed by big pharma": "Classic conspiracy narrative used to explain lack of evidence.",
    "secret government": "Vague references to 'secret' authorities are red flags.",
    "microchip": "Commonly linked to debunked vaccine-related conspiracy theories.",
    "shocking discovery": "Clickbait language designed to trigger emotional responses."
}

def detect_suspicious(text):
    highlighted = text
    explanations = []
    for phrase, explanation in suspicious_patterns.items():
        if phrase.lower() in text.lower():
            pattern = re.compile(re.escape(phrase), re.IGNORECASE)
            highlighted = pattern.sub(f'<span class="suspicious-highlight">{phrase}</span>', highlighted)
            if explanation not in explanations:
                explanations.append(explanation)
    return highlighted, explanations

# NLP / Model Setup
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
    
    # Mapping logic: ensure we handle both Label IDs and Strings
    idx = torch.argmax(probs).item()
    label = model.config.id2label[idx].upper() 
    
    # Standardize label names if the model uses LABEL_0/LABEL_1
    if "0" in label or "FAKE" in label:
        label = "FAKE"
    else:
        label = "REAL"
        
    return label, round(confidence * 100, 2)

@app.route("/", methods=["GET", "POST"])
def index():
    global articles_analyzed, fake_count, real_count
    
    # 1. Initialize data dictionary
    data = {
        "prediction": None, 
        "confidence": None, 
        "claims_data": [], 
        "entities": [], 
        "highlighted_text": None, 
        "explanation_points": [], 
        "original_text": None
    }

    # 2. Logic for Form Submission
    if request.method == "POST":
        text = request.form.get("news_text")
        if text and text.strip():
            data["original_text"] = text
            data["prediction"], data["confidence"] = classify_text(text)
            
            articles_analyzed += 1
            if data["prediction"] == "FAKE": 
                fake_count += 1
            else: 
                real_count += 1
                
            data["highlighted_text"], data["explanation_points"] = detect_suspicious(text)
            doc = nlp(text)
            data["entities"] = [(ent.text, ent.label_) for ent in doc.ents]

    # 3. Prepare variables for the template
    uptime = int(time.time() - start_time)
    
    metrics_summary = {
        "analyzed": articles_analyzed,
        "distribution": f"F:{fake_count} | R:{real_count}",
        "session": f"{uptime}s"
    }
    
    # FIX: Define the 'deployment' variable that index.html is looking for
    deployment_info = {
        "status": "Active (Local)",
        "provider": "Flask Dev Server"
    }

    # 4. Pass all variables to render_template
    return render_template(
        "index.html", 
        **data, 
        metrics=metrics_summary, 
        deployment=deployment_info
    )

if __name__ == "__main__":
    app.run(debug=True)