import json
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE = Path(__file__).parent
model = joblib.load(BASE / "glycolens_model.joblib")
THRESHOLD = json.load(open(BASE / "glycolens_config.json"))["threshold"]

app = Flask(__name__)

# name: (label, min, max, type). Zero is allowed for Insulin / SkinThickness (= unknown).
FIELDS = {
    "Pregnancies": ("Pregnancies", 0, 20, int),
    "Glucose": ("Glucose (mg/dL)", 40, 300, float),
    "BloodPressure": ("Blood pressure (mm Hg)", 30, 140, float),
    "SkinThickness": ("Skin thickness (mm, 0 if unknown)", 0, 100, float),
    "Insulin": ("Insulin (mu U/ml, 0 if unknown)", 0, 900, float),
    "BMI": ("BMI", 10, 70, float),
    "DiabetesPedigreeFunction": ("Diabetes pedigree function", 0.05, 2.5, float),
    "Age": ("Age (years)", 15, 100, int),
}


def parse(form):
    """Validate input and return (record, errors)."""
    record, errors = {}, []
    for key, (label, lo, hi, typ) in FIELDS.items():
        raw = (form.get(key) or "").strip()
        try:
            val = typ(float(raw))
        except ValueError:
            errors.append(f"{label}: please enter a number.")
            continue
        if not lo <= val <= hi:
            errors.append(f"{label}: must be between {lo} and {hi}.")
            continue
        record[key] = val
    return record, errors


def predict(record):
    p = float(model.predict_proba(pd.DataFrame([record]))[:, 1][0])
    return {
        "probability": round(p, 4),
        "percent": round(p * 100, 1),
        "high_risk": p >= THRESHOLD,
        "label": "High risk" if p >= THRESHOLD else "Low risk",
    }


@app.route("/", methods=["GET", "POST"])
def index():
    result, errors, values = None, [], {}
    if request.method == "POST":
        values = request.form.to_dict()
        record, errors = parse(request.form)
        if not errors:
            result = predict(record)
    return render_template("index.html", fields=FIELDS, result=result, errors=errors, values=values)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True) or {}
    record, errors = parse(data)
    if errors:
        return jsonify({"errors": errors}), 400
    return jsonify(predict(record))


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True)
