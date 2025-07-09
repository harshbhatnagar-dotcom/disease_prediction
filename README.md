# 🩺 Symptom2Disease - Multi-Disease Prediction from Symptoms

This Flask-based web app uses machine learning to predict **up to 3 likely diseases** based on user-described symptoms. It also suggests relevant medical tests for each predicted disease.

---

## 🚀 Features

- 📝 Accepts natural language symptom descriptions
- 🤖 Predicts multiple diseases using a trained ML model
- 🧪 Recommends medical tests based on predictions
- 🎨 Clean UI with Bootstrap 5
- ☁️ Deployable on platforms like Render or Replit

---

## 🧠 Machine Learning

- **Model**: `OneVsRestClassifier` with `MultinomialNB`
- **Vectorizer**: `TfidfVectorizer` (bi- & tri-grams)
- **Label Encoder**: `MultiLabelBinarizer`
- **Trained on**: A curated dataset with symptoms and multiple disease labels

---
[Live Demo](https://disease-prediction-qcwq.onrender.com)

