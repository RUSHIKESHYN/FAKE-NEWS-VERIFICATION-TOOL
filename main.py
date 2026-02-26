from fastapi import FastAPI,Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import re

app= FastAPI()
templates=Jinja2Templates(directory="templates")

model,vectorizer = joblib.load("model/model.pkl")

def clean_text(text):
    text+text.lower()
    text=re.sub(r'[^a-z]','',text)
    return text

@app.get("/",response_class=HTMLResponse)
def home(request:Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict",response_class=HTMLResponse)
def predict(request:Request,text: str=Form(...)):

    cleaned = clean_text(text)

    vec=vectorizer.transform([cleaned])
    result=model.predict(vec)[0]
    prob=model.predict_proba(vec)[0]

    return templates.TemplateResponse("index.html", {
        "request":request,
        "prediction": "REAL NEWS" if result == 1 else "FAKE NEWS",
        "fake":round(prob[0]*100,2),
        "real":round(prob[1]*100,2),
        "input":text

    })