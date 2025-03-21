from flask import Flask, request, jsonify, Response
import xgboost as xgb
import pandas as pd
import logging
import hashlib
from datetime import datetime
from prometheus_client import Counter, Histogram, Gauge, generate_latest

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load the trained XGBoost fraud detection model
model = xgb.XGBClassifier()
model.load_model("./asset/xgboost_fraud_model.json")

# Define expected feature names (as per model training)
feature_mapping = {
    "transaction_payment_mode": "transaction_payment_mode_anonymous",
    "payment_gateway_bank": "payment_gateway_bank_anonymous",
    "payer_browser": "payer_browser_anonymous",
    "payer_email": "payer_email_anonymous",
    "payee_ip": "payee_ip_anonymous",
    "payer_mobile": "payer_mobile_anonymous",
    "transaction_id": "transaction_id_anonymous",
    "payee_id": "payee_id_anonymous"
}

# Prometheus Metrics
fraud_predictions = Counter("fraud_predictions_total", "Total number of fraud predictions", ["is_fraud_predicted"])
fraud_reports = Counter("fraud_reports_total", "Total number of reported frauds", ["is_fraud_reported"])
transactions_total = Counter("transactions_total", "Total number of transactions")
fraud_trend = Gauge("fraud_trend", "Fraud trend over time")
request_latency = Histogram("request_latency_seconds", "Latency of API requests")

# Function to hash categorical string values
def hash_string(value):
    return int(hashlib.sha256(value.encode()).hexdigest(), 16) % (10**9)  # Convert to 9-digit integer

@app.route("/metrics")
def metrics():
    return Response(generate_latest(), mimetype="text/plain")

@app.route("/predict", methods=["POST"])
def predict():
    with request_latency.time():
        try:
            # Parse incoming JSON data
            data = request.get_json()
            
            # Convert transaction_amount to float
            try:
                data["transaction_amount"] = float(data["transaction_amount"])
            except ValueError:
                logging.error("Invalid transaction_amount format")
                return jsonify({"error": "Invalid transaction_amount format. Must be a number."}), 400
            
            # Extract and convert transaction date
            try:
                transaction_date = datetime.strptime(data["transaction_date"], "%m/%d/%Y %H:%M")
                data["transaction_day"] = transaction_date.day
                data["transaction_month"] = transaction_date.month
                data["transaction_year"] = transaction_date.year
            except Exception as e:
                logging.error(f"Invalid transaction_date format: {str(e)}")
                return jsonify({"error": "Invalid transaction_date format. Use MM/DD/YYYY HH:MM"}), 400
            
            # Hash transaction_channel
            data["transaction_channel"] = hash_string(str(data["transaction_channel"]))
            
            # Rename and hash categorical features to match model training data
            transformed_data = {}
            transformed_data["transaction_amount"] = data["transaction_amount"]
            transformed_data["transaction_channel"] = data["transaction_channel"]
            transformed_data["transaction_day"] = data["transaction_day"]
            transformed_data["transaction_month"] = data["transaction_month"]
            transformed_data["transaction_year"] = data["transaction_year"]
            
            for original, renamed in feature_mapping.items():
                if original in data:
                    transformed_data[renamed] = hash_string(str(data[original]))
                else:
                    logging.warning(f"Missing feature: {original}")
                    transformed_data[renamed] = hash_string("unknown")
            
            # Define model input feature order
            feature_order = [
                "transaction_amount", "transaction_channel", "transaction_payment_mode_anonymous", 
                "payment_gateway_bank_anonymous", "payer_browser_anonymous", "payer_email_anonymous", "payee_ip_anonymous", 
                "payer_mobile_anonymous", "transaction_id_anonymous", "payee_id_anonymous", "transaction_day", 
                "transaction_month", "transaction_year"
            ]
            
            # Convert data into DataFrame
            df = pd.DataFrame([{key: transformed_data[key] for key in feature_order}])
            
            # Make fraud prediction
            prediction = model.predict(df)[0]
            probability = model.predict_proba(df)[0][1]  # Probability of fraud
            
            # Update Prometheus Metrics
            fraud_predictions.labels(is_fraud_predicted=str(prediction)).inc()
            transactions_total.inc()
            fraud_trend.set(probability)

            response = {
                "is_fraud": int(prediction),
                "fraud_probability": float(probability),
                "transaction_id": transformed_data.get("transaction_id_anonymous", "unknown")
            }
            
            logging.info(f"Prediction Response: {response}")
            return jsonify(response)
        
        except Exception as e:
            logging.error(f"Error processing request: {str(e)}")
            return jsonify({"error": str(e)}), 400

# Define a Prometheus counter for reported frauds
reported_frauds = Counter("reported_frauds", "Total number of reported frauds")

@app.route("/report_fraud", methods=["POST"])
def report_fraud():
    """Endpoint to report a fraud case."""
    try:
        data = request.get_json()
        if data.get("is_fraud_reported", False):
            reported_frauds.inc()  # Increment fraud counter
        return jsonify({"message": "Fraud reported"}), 200
    except Exception as e:
        logging.error(f"Error reporting fraud: {str(e)}")
        return jsonify({"error": str(e)}), 400

# Run the API
if __name__ == "__main__":
    logging.info("Starting Fraud Detection API...")
    app.run(host="0.0.0.0", port=5000, debug=True)
