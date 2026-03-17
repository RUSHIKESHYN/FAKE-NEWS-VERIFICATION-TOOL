FAKE NEWS DETECTION AND VERIFICATION

This project provides an AI-powered tool for analyzing news content and classifying it as Real or Fake. It includes explainability features, a trusted sources system, and persistence via a MySQL database.

🚀 Features
News Analysis: Submit a headline, full article, and URL to receive a prediction.
Explainable Output: Highlights suspicious phrases, shows model confidence, and displays SHAP/LIME feature contributions.
Admin Dashboard: (Removed) — The app focuses on single-page analysis.
Storage: All submissions and predictions are stored in MySQL.
Containerized: Run locally with Docker (or deploy to cloud platforms).
🧩 Tech Stack
Backend / UI: Python + Streamlit
Database: MySQL (via SQLAlchemy)
Explainability: Keyword-based suspicious phrase detection

🗂 Database Schema
The system includes the following tables:
users - admin and user information
news_articles - submitted articles
predictions - model predictions and explanations
trusted_sources - verified news sources
system_metrics - analytics and uptime

🔍 How the Model Works
This prototype uses a TF-IDF + Logistic Regression model trained on a small sample dataset.
Detects suspicious phrases (e.g., "miracle cure", "suppressed by government").
Uses trusted sources to reduce false positives.
Shows SHAP/LIME feature contributions for explainability.
You can extend the model with a larger dataset, use Transformers, or add more advanced explainability.

✅ Next Improvements
Replace heuristic classifier with a trained model (now implemented via TF-IDF + Logistic Regression)
Add user authentication and improved dashboard UX (login/logout, per-user history)
Store ground-truth labels + compute real accuracy
Add API endpoints for integration
Add unit tests
