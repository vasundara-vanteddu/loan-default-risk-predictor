from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load saved model components
model = joblib.load("model/loan_model.pkl")
scaler = joblib.load("model/scaler.pkl")
selector = joblib.load("model/selector.pkl")

# Home route (serves frontend)
@app.route("/")
def home():
    return render_template("index.html")

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json["features"]

        features = np.array(data).reshape(1, -1)

        # Apply same preprocessing steps
        features_scaled = scaler.transform(features)
        features_selected = selector.transform(features_scaled)

        prediction = model.predict(features_selected)[0]
        probability = model.predict_proba(features_selected)[0][1]

        return jsonify({
            "prediction": int(prediction),
            "default_probability": float(probability)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
    
    import os

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))