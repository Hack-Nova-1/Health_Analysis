"""
Disease Prediction Backend API
----------------------------------
Flask backend with:
✅ Model inference
✅ Precaution retrieval
✅ SQLite logging
✅ CORS support for frontend integration
"""
from difflib import get_close_matches
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
from datetime import datetime
import os

# -------------------------
# Flask App Initialization
# -------------------------
app = Flask(__name__)
CORS(app)  # enable CORS for all domains

# -------------------------
# Model and Data Loading
# -------------------------
MODEL_PATH = "models/final_best_GradientBoosting_pipeline.pkl"
ENCODER_PATH = "models/label_encoder.pkl"
PRECAUTION_PATH = "Disease precaution.csv"

model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)
precautions_df = pd.read_csv(PRECAUTION_PATH, comment='#')

# -------------------------
# SQLite Setup
# -------------------------
DB_PATH = "disease_predictions.db"

def init_db():
    """Initialize SQLite DB for storing predictions"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            age REAL,
            gender INTEGER,
            heart_rate REAL,
            body_temp REAL,
            oxygen_sat REAL,
            bp_systolic REAL,
            bp_diastolic REAL,
            symptoms TEXT,
            predicted_disease TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

def log_prediction(data_dict, disease_name):
    """Insert a prediction record into the DB"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions (
            timestamp, age, gender, heart_rate, body_temp, oxygen_sat,
            bp_systolic, bp_diastolic, symptoms, predicted_disease
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        data_dict.get("Age"), data_dict.get("Gender"),
        data_dict.get("Heart_Rate_bpm"), data_dict.get("Body_Temperature_C"),
        data_dict.get("Oxygen_Saturation_%"), data_dict.get("BP_Systolic"),
        data_dict.get("BP_Diastolic"),
        " ".join([data_dict.get("Symptom_1", ""), data_dict.get("Symptom_2", ""), data_dict.get("Symptom_3", "")]),
        disease_name
    ))
    conn.commit()
    conn.close()

# -------------------------
# Helper Functions
# -------------------------
def get_precautions(disease_name):
    """
    Retrieve precautions for a given disease, using fuzzy matching.
    This ensures close matches (e.g. 'Flu' ↔ 'Influenza') are found.
    """
    if not isinstance(disease_name, str) or not disease_name.strip():
        return []

    # Normalize input and dataset
    disease_name = disease_name.strip().lower()
    all_diseases = precautions_df["Disease"].dropna().astype(str).str.lower().tolist()

    # Try direct match first
    if disease_name in all_diseases:
        row = precautions_df.loc[precautions_df["Disease"].str.lower() == disease_name]
    else:
        # Fuzzy match to handle similar names
        close_matches = get_close_matches(disease_name, all_diseases, n=1, cutoff=0.6)
        if close_matches:
            row = precautions_df.loc[precautions_df["Disease"].str.lower() == close_matches[0]]
        else:
            row = pd.DataFrame()

    if row.empty:
        return ["No specific precautions found. Please consult a doctor."]

    # Collect all precaution columns dynamically
    precautions = [
        str(v).strip()
        for v in row.iloc[0, 1:].dropna().values.tolist()
        if str(v).strip() not in ["", "nan"]
    ]
    return precautions if precautions else ["General care: rest, hydrate, monitor symptoms."]
# -------------------------
# Routes
# -------------------------
@app.route("/")
def home():
    return jsonify({"message": "Disease Prediction API with DB Logging & CORS is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input provided"}), 400

        df = pd.DataFrame([data])
        if "All_Symptoms" not in df.columns:
            df["All_Symptoms"] = (
                df.get("Symptom_1", "") + " " +
                df.get("Symptom_2", "") + " " +
                df.get("Symptom_3", "")
            ).str.lower()

        for col in ["Avg_Symptom_Severity", "High_Severity_Count"]:
            if col not in df.columns:
                df[col] = 0

        # Predict disease
        prediction = model.predict(df)[0]
        disease = label_encoder.inverse_transform([prediction])[0]
        precautions = get_precautions(disease)

        # Log prediction in SQLite
        log_prediction(data, disease)

        return jsonify({
            "predicted_disease": disease,
            "precautions": precautions
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/history", methods=["GET"])
def history():
    """Return last N predictions from the database"""
    try:
        n = int(request.args.get("limit", 10))
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (n,))
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        conn.close()
        return jsonify([dict(zip(columns, row)) for row in rows])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------------
# Run App
# -------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

