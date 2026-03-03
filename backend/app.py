from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# Load saved model components
model = joblib.load("model/loan_model.pkl")
scaler = joblib.load("model/scaler.pkl")
selector = joblib.load("model/selector.pkl")

# Get feature names from scaler (important for production)
feature_names = scaler.feature_names_in_

# Home route (serves frontend)
@app.route("/")
def home():
    return render_template("index.html")

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json["features"]

        # Convert to DataFrame with correct column names
        features_df = pd.DataFrame([data], columns=feature_names)

        # Apply preprocessing
        features_scaled = scaler.transform(features_df)
        features_selected = selector.transform(features_scaled)

        prediction = model.predict(features_selected)[0]
        probability = model.predict_proba(features_selected)[0][1]

        return jsonify({
            "prediction": int(prediction),
            "default_probability": float(probability)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# Production run
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))