from flask import Flask, request, jsonify
import joblib
import pandas as pd
import time

# Initialize Flask app
app = Flask(__name__)

# Load model and features
MODEL_PATH = "/mnt/f/Zaalima Internship/Zaalima Project/factoryguard-ai/output/final_model.pkl"
FEATURES_PATH = "/mnt/f/Zaalima Internship/Zaalima Project/factoryguard-ai/output/features.pkl"

final_model = joblib.load(MODEL_PATH)
FEATURES = joblib.load(FEATURES_PATH)

# Get exact feature names used during training to avoid feature mismatch

# Health check route
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "API is running",
        "model": "LightGBM Predictive Maintenance Model"
    })

# Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    try:
        start=time.perf_counter()
        data = request.get_json()

        if data is None:
            return jsonify({"error": "Invalid JSON input"}), 400

        # Convert input JSON to DataFrame
        df = pd.DataFrame([data])

        # Ensure all required features exist
        missing_features = [f for f in FEATURES if f not in df.columns]

        if missing_features:
            return jsonify({
                "error": "Missing required features",
                "missing_features": missing_features
            }), 400

        # Predict probability (failure = class 1)
        failure_prob = final_model.predict_proba(df)[:,1][0]

        def categorize_risk(score):
            if score >= 0.85:
                return "CRITICAL"
            elif score >= 0.60:
                return "HIGH"
            elif score >= 0.30:
                return "MEDIUM"
            else:
                return "LOW"
        

        return jsonify({
            "failure_probability": round(float(failure_prob), 4),
            "risk_level": categorize_risk(failure_prob),
            "Server Latency":round((time.perf_counter()-start)*1000,2)
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500
    
@app.route("/features", methods=["GET"])
def features():
    return jsonify({
        "required_features": FEATURES
    })


# Run Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003, debug=False)
