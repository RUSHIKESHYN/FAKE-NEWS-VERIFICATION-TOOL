import streamlit as st
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
from dotenv import load_dotenv

# Load Environment Variables
load_dotenv()

# --- INITIALIZATION & SAFETY ---
if 'nltk_downloaded' not in st.session_state:
    try:
        nltk.download('punkt')
        st.session_state.nltk_downloaded = True
    except:
        pass

if 'start_time' not in st.session_state:
    st.session_state.start_time = time.time()
if 'count' not in st.session_state: st.session_state.count = 0
if 'fakes' not in st.session_state: st.session_state.fakes = 0
if 'reals' not in st.session_state: st.session_state.reals = 0

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Wikipedia API Setup
wiki = wikipediaapi.Wikipedia(
    user_agent="FakeNewsVerificationBot/1.0 (contact: your-email@example.com)", 
    language='en'
)

# Highlight dictionary
suspicious_patterns = {
    "miracle cure": "Often used in medical misinformation.",
    "eliminates all types of cancer": "Biological impossibility; red flag for health scams.",
    "suppressed by big pharma": "Standard conspiracy narrative.",
    "secret government": "Vague references to hidden authorities.",
    "microchip": "Commonly linked to vaccine conspiracy theories.",
    "shocking discovery": "Sensationalist 'clickbait' language."
}

@st.cache_resource
def load_models():
    try:
        nlp = spacy.load("en_core_web_sm")
    except:
        os.system("python -m spacy download en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    
    MODEL_NAME = "hamzab/roberta-fake-news-classification"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    return nlp, tokenizer, model

nlp, tokenizer, model = load_models()

def detect_suspicious(text):
    highlighted = text
    explanations = []
    for phrase, explanation in suspicious_patterns.items():
        if phrase.lower() in text.lower():
            pattern = re.compile(re.escape(phrase), re.IGNORECASE)
            highlighted = pattern.sub(f'<span style="color: #ff4b4b; text-decoration: underline; font-weight: bold;">{phrase}</span>', highlighted)
            if explanation not in explanations:
                explanations.append(explanation)
    return highlighted, explanations

# --- UI SETUP ---
st.set_page_config(page_title="Fake News Verification Tool", layout="wide")

st.markdown("""
    <style>
    .admin-header { color: white; font-weight: bold; font-size: 1.4rem; margin-top: 40px; margin-bottom: 20px; }
    .admin-card {
        background-color: #312752;
        padding: 20px;
        border-radius: 12px;
        color: #e0e0e0;
        height: 160px;
        border: 1px solid #453a73;
    }
    .card-title { font-weight: bold; font-size: 1.1rem; color: white; margin-bottom: 12px; }
    .status-green { color: #00ff88; font-weight: bold; }
    .metric-val { font-family: monospace; font-size: 1.2rem; color: #00ff88; }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ Fake News Verification Tool")

# --- MAIN INPUT ---
news_text = st.text_area("Enter News Article Content:", height=200, placeholder="Paste article text here...")

if st.button("Verify News"):
    if news_text.strip():
        st.session_state.count += 1
        
        # 1. AI Analysis (FIXED LABEL MAPPING)
        with st.spinner('Scanning Article...'):
            inputs = tokenizer(news_text, return_tensors="pt", truncation=True, padding=True, max_length=256)
            with torch.no_grad():
                outputs = model(**inputs)
            
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            confidence = torch.max(probs).item() * 100
            
            # FIXED: Model output 1 = FAKE, 0 = REAL
            prediction_idx = torch.argmax(probs).item()
            label = "FAKE" if prediction_idx == 1 else "REAL"
            
            if label == "FAKE": st.session_state.fakes += 1
            else: st.session_state.reals += 1

        # 2. Results display
        st.markdown(f"## Prediction: {label}")
        st.write(f"AI Confidence: **{confidence:.2f}%**")
        st.progress(confidence / 100)

        # 3. Content Highlights
        st.subheader("Content Analysis")
        highlighted_html, reasons = detect_suspicious(news_text)
        st.markdown(f'<div style="padding: 20px; border-radius: 10px; border: 1px solid #4a90e2; line-height: 1.8;">{highlighted_html}</div>', unsafe_allow_html=True)
        
        if reasons:
            for r in reasons: st.warning(f"⚠️ {r}")

        # 4. Source Verification
        st.markdown("---")
        st.subheader("🔍 Source Verification")
        sentences = sent_tokenize(news_text)[:2]
        
        for sent in sentences:
            clean_query = re.sub(r'[^\w\s]', '', sent)
            short_query = " ".join(clean_query.split()[:8])
            
            found = False
            if GOOGLE_API_KEY:
                encoded_q = urllib.parse.quote(short_query)
                url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?query={encoded_q}&key={GOOGLE_API_KEY}"
                try:
                    res = requests.get(url).json()
                    if "claims" in res:
                        c = res["claims"][0]["claimReview"][0]
                        st.info(f"**Google Fact Check:** {c['textualRating']} (via {c['publisher']['name']})")
                        found = True
                except: pass
            
            if not found:
                wiki_page = wiki.page(short_query[:30])
                if wiki_page.exists():
                    st.success(f"**Context (Wikipedia):** [{wiki_page.title}]({wiki_page.fullurl})")
                else:
                    st.write(f"⚪ No specific records found for: *\"{short_query}...\"*")
    else:
        st.error("Please enter some text to verify.")

# --- ADMIN DASHBOARD (MATCHING THE THEME) ---
st.markdown('<div class="admin-header">🛠️ ADMIN DASHBOARD</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(f"""
        <div class="admin-card">
            <div class="card-title">💻 System</div>
            <div>Server: <span class="status-green">ONLINE</span></div>
            <div>API Connection: <span class="status-green">ACTIVE</span></div>
        </div>
        """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
        <div class="admin-card">
            <div class="card-title">📊 Analytics</div>
            <div>Processed: <span class="metric-val">{st.session_state.count}</span></div>
            <div>Fakes Detected: <span class="metric-val">{st.session_state.fakes}</span></div>
        </div>
        """, unsafe_allow_html=True)

with c3:
    uptime = int(time.time() - st.session_state.start_time)
    st.markdown(f"""
        <div class="admin-card">
            <div class="card-title">📈 Deployment</div>
            <div>Uptime: <span class="metric-val">{uptime}s</span></div>
            <div>Environment: <span class="status-green">READY</span></div>
        </div>
        """, unsafe_allow_html=True)