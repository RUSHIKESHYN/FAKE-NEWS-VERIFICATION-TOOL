from flask import Flask, render_template, request
import spacy
import re
import joblib
from init_db import init_db, insert_processed_text, insert_entity, insert_claim


# -------------------------
# Create Flask App FIRST
# -------------------------
app = Flask(__name__)


# -------------------------
# Load Models
# -------------------------
nlp = spacy.load("en_core_web_sm")
model, vectorizer = joblib.load("model/model.pkl")


# -------------------------
# Initialize Database
# -------------------------
init_db()


# -------------------------
# Text Cleaning Function
# -------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text.strip()


# -------------------------
# Simple Verification Logic
# -------------------------
def verify_claim(claim):
    suspicious_words = ["100%", "always", "never", "guaranteed"]

    for word in suspicious_words:
        if word.lower() in claim.lower():
            return "Suspicious"

    return "Needs Verification"


# -------------------------
# Main Route
# -------------------------
@app.route("/", methods=["GET", "POST"])
def index():

    # Define variables BEFORE POST
    prediction = None
    confidence = None
    claims = []
    entities = []
    cleaned_text = ""
    text = ""   # VERY IMPORTANT (prevents UnboundLocalError)

    if request.method == "POST":
        text = request.form.get("text")

        if text and text.strip():

            # spaCy Processing
            doc = nlp(text)

            # Clean text
            cleaned_text = clean_text(text)

            # Vectorize
            vectorized_text = vectorizer.transform([text])

            # Predict
            pred = model.predict(vectorized_text)[0]
            prob = model.predict_proba(vectorized_text).max()

            prediction = "REAL" if pred == 1 else "FAKE"
            confidence = round(prob * 100, 2)

            # Extract Entities
            entities = [(ent.text, ent.label_) for ent in doc.ents]

            # Extract Claims
            raw_claims = [sent.text for sent in doc.sents]

            # Save to DB
            text_id = insert_processed_text(
                text,
                cleaned_text,
                prediction,
                confidence
            )

            for ent_text, ent_label in entities:
                insert_entity(text_id, ent_text, ent_label)

            claims = []
            for claim in raw_claims:
                verification = verify_claim(claim)
                insert_claim(text_id, claim, verification)
                claims.append((claim, verification))

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        claims=claims,
        entities=entities,
        original_text=text
    )


# -------------------------
# Run App
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)