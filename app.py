from flask import Flask,request,render_template
import joblib
import pandas as pd
import os

app=Flask(__name__)
model=joblib.load("model.pkl")
mlb=joblib.load("label.pkl")

base_dir = os.path.dirname(__file__)
# Load Recommended Tests
tests_path = os.path.join(base_dir, "data", "Disease_Diagnostic_Tests.csv")
test_df = pd.read_csv(tests_path)
test_map = dict(zip(test_df["Disease"], test_df["Recommended Tests"]))

# Load Precautions
precautions_path = os.path.join(base_dir, "data", "Disease_Precautions.csv")
precautions_df = pd.read_csv(precautions_path)
precautions_map = dict(zip(precautions_df["Disease"], precautions_df["Precautions"]))

# Load Causes
causes_path = os.path.join(base_dir, "data", "Disease_Causes.csv")
causes_df = pd.read_csv(causes_path)
causes_map = dict(zip(causes_df["Disease"], causes_df["Causes"]))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    symptoms = request.form["symptoms"]
    input_df = pd.DataFrame({"text": [symptoms]})
    probs = model.predict_proba(input_df)[0]
    top3 = probs.argsort()[-3:][::-1]
    predictions = [mlb.classes_[i] for i in top3]

    
    test_info = {}
    precaution_info = {}
    cause_info = {}

    for disease in predictions:
        test_info[disease] = test_map.get(disease, "No test info available")
        precaution_info[disease] = precautions_map.get(disease, "No precaution info available")
        cause_info[disease] = causes_map.get(disease, "No cause info available")

    return render_template(
        "predict.html",
        symptoms=symptoms,
        predictions=predictions,
        test_info=test_info,
        precaution_info=precaution_info,
        cause_info=cause_info
    )


