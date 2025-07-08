from flask import Flask,request,render_template
import joblib
import pandas as pd
import os

app=Flask(__name__)
model=joblib.load("model.pkl")
mlb=joblib.load("label.pkl")

csv_path = os.path.join(os.path.dirname(__file__), "data", "Disease_Test_Recommendations.csv")

# Load the test mapping
test_df = pd.read_csv(csv_path)
test_map = dict(zip(test_df["Disease"], test_df["Recommended Tests"]))

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

    for disease in predictions:
        if disease in test_map:
            test_info[disease] = test_map[disease]
        else:
            test_info[disease] = "No test info available"
    return render_template("predict.html", symptoms=symptoms, predictions=predictions,test_info=test_info)


if __name__ =="__main__":
    app.run(debug=True)