import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load the trained ML model pipeline on startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"ML model file not found at {MODEL_PATH}. Please run train_model.py first.")

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print("Successfully loaded trained ML model pipeline.")
except Exception as e:
    raise RuntimeError(f"Error loading model from {MODEL_PATH}: {str(e)}")

@app.route('/')
def home():
    """Renders the dashboard/landing page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Accepts patient health attributes, runs ML prediction, and returns
    predicted risk status and probability percentage.
    Supports both JSON payloads and standard form submissions.
    """
    try:
        # Determine format (JSON or URL Form Encoded)
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form

        # Extract and validate fields
        try:
            age = int(data.get('age'))
            sex = int(data.get('sex'))
            cp = int(data.get('cp'))
            trestbps = int(data.get('trestbps'))
            chol = int(data.get('chol'))
            thalach = int(data.get('thalach'))
        except (TypeError, ValueError) as e:
            return jsonify({
                'success': False,
                'error': 'Invalid input data. All fields must be numeric.'
            }), 400

        # Construct input DataFrame with exact features names as trained on
        # Columns: ['age', 'sex', 'cp', 'trestbps', 'chol', 'thalach']
        features = pd.DataFrame([{
            'age': age,
            'sex': sex,
            'cp': cp,
            'trestbps': trestbps,
            'chol': chol,
            'thalach': thalach
        }])

        # Perform prediction
        # model is a pipeline containing StandardScaler and LogisticRegression
        prediction = int(model.predict(features)[0])
        # Get probability of class 1 (heart disease risk)
        probability = float(model.predict_proba(features)[0][1])

        # Formulate a dynamic, friendly recommendation response
        risk_detected = prediction == 1
        probability_percentage = round(probability * 100, 1)

        # Generate a personalized clinical guidance message
        bullet_points = []
        if trestbps > 130:
            bullet_points.append("Elevated resting blood pressure (>130 mmHg)")
        if chol > 200:
            bullet_points.append("High serum cholesterol level (>200 mg/dl)")
        if thalach < 120:
            bullet_points.append("Reduced maximum heart rate capacity (<120 bpm)")

        if risk_detected:
            title_msg = "Heart Disease Risk Detected"
            if len(bullet_points) > 0:
                body_msg = f"Patient shows indicators of cardiovascular risk, particularly: {', '.join(bullet_points)}. We strongly recommend consulting a cardiologist for a comprehensive clinical evaluation."
            else:
                body_msg = "The analysis indicates a elevated risk of cardiovascular issues. A standard checkup is recommended."
        else:
            title_msg = "No Heart Disease Risk Detected"
            if len(bullet_points) > 0:
                body_msg = f"Overall risk is low, but the patient displays mild anomalies: {', '.join(bullet_points)}. Maintain a healthy diet and active lifestyle."
            else:
                body_msg = "All metabolic and cardiovascular markers appear to be in optimal ranges. Keep up the healthy habits!"

        return jsonify({
            'success': True,
            'risk_detected': risk_detected,
            'probability': probability_percentage,
            'title': title_msg,
            'message': body_msg
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f"An error occurred during prediction: {str(e)}"
        }), 500

if __name__ == '__main__':
    # Running locally
    app.run(debug=True, port=5000)
