from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

# Load ML models
iso_model = joblib.load("chemguard_stage1_anomaly.pkl")
sev_model = joblib.load("chemguard_stage2_severity.pkl")

app = Flask(__name__)

@app.route("/")
def home():
    return "ChemGuard ML API is running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        df = pd.DataFrame([{
            "gas": data["gas"],
            "temperature": data["temperature"],
            "humidity": data["humidity"],
            "pressure": data["pressure"],
            "flame": data["flame"],
            "gas_delta": data.get("gas_delta", 0),
            "temp_delta": data.get("temp_delta", 0)
        }])

        # Stage 1: Anomaly detection
        anomaly = iso_model.predict(df)[0]

        if anomaly == 1:
            return jsonify({
                "risk": "SAFE",
                "stage": "NORMAL"
            })

        # Stage 2: Severity classification
        severity = sev_model.predict(df)[0]
        risk = "WARNING" if severity == 1 else "CRITICAL"

        return jsonify({
            "risk": risk,
            "stage": "ANOMALY"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)