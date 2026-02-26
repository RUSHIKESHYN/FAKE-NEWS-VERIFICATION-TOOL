import joblib

model, vectorization =joblib.load("model/model.pkl")

while True:
    text = input("\nEnter the news (type exit to stop):")
    if text.lower() =="exit":
        break
    text_vector=vectorization.transform([text])
    result=model.predict(text_vector)[0]
    prob=model.predict_proba(text_vector)[0]

    fake_confidence=round(prob[0]*100,2)
    real_confidence=round(prob[1]*100,2)

    print("\nPrediction Result:")

    if result == 1:
        print("REAl")
    else: print("FAKE")

    print(f"Fake Confidence: {fake_confidence}")
    print(f"Real confidence: {real_confidence}")