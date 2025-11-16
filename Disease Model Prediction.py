"""
Disease Prediction System – Multi-Dataset Hybrid Tuning Pipeline
Author: Data Analyst (ChatGPT-5)

Required files in the same folder:
  1. disease_diagnosis.csv
  2. Symptom-severity.csv
  3. Disease precaution.csv
"""

import pandas as pd, numpy as np, time, joblib, os
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import randint as sp_randint, uniform as sp_uniform

# ---------------------------------------------------------------------
# 1️⃣  Load & clean all datasets
# ---------------------------------------------------------------------
print("Loading datasets...")
df_main = pd.read_csv("disease_diagnosis.csv")
df_severity = pd.read_csv("Symptom-severity.csv")
df_precautions = pd.read_csv("Disease precaution.csv")

# Clean main dataset
bp_split = df_main["Blood_Pressure_mmHg"].astype(str).str.split("/", expand=True)
df_main["BP_Systolic"] = pd.to_numeric(bp_split[0], errors='coerce').fillna(0)
df_main["BP_Diastolic"] = pd.to_numeric(bp_split[1], errors='coerce').fillna(0)
df_main["All_Symptoms"] = (
    df_main["Symptom_1"].fillna('') + " " +
    df_main["Symptom_2"].fillna('') + " " +
    df_main["Symptom_3"].fillna('')
).str.lower()
df_main["Gender"] = df_main["Gender"].map({"Male": 1, "Female": 0}).fillna(0).astype(int)
df_main.drop(columns=["Patient_ID", "Treatment_Plan", "Blood_Pressure_mmHg"], inplace=True, errors='ignore')

# Severity mapping
df_severity['Symptom'] = df_severity['Symptom'].str.lower().str.strip()
severity_map = dict(zip(df_severity['Symptom'], df_severity['weight']))

def compute_severity(row):
    symptoms = [row.get('Symptom_1',''), row.get('Symptom_2',''), row.get('Symptom_3','')]
    weights = [severity_map.get(str(s).lower(), 0) for s in symptoms]
    valid = [w for w in weights if w>0]
    avg = np.mean(valid) if valid else 0.0
    high_count = sum(w >= 3 for w in valid)
    return pd.Series({'Avg_Symptom_Severity': avg, 'High_Severity_Count': high_count})

severity_features = df_main.apply(compute_severity, axis=1)
df_main = pd.concat([df_main, severity_features], axis=1)

# ---------------------------------------------------------------------
# 2️⃣  Feature definition & preprocessing
# ---------------------------------------------------------------------
text_col = "All_Symptoms"
numeric_cols = [
    'Age', 'Gender', 'Heart_Rate_bpm', 'Body_Temperature_C',
    'Oxygen_Saturation_%', 'BP_Systolic', 'BP_Diastolic',
    'Avg_Symptom_Severity', 'High_Severity_Count'
]

X = df_main[[text_col] + numeric_cols]
y = df_main["Diagnosis"]

le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

preprocessor = ColumnTransformer([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=1500), text_col),
    ('num', StandardScaler(), numeric_cols)
])

# ---------------------------------------------------------------------
# 3️⃣  Define search grids
# ---------------------------------------------------------------------
grids = {
    "Logistic": {
        'clf__C': [0.01, 0.1, 1, 10],
        'clf__solver': ['lbfgs']
    },
    "RandomForest": {
        'clf__n_estimators': sp_randint(100, 500),
        'clf__max_depth': [None, 10, 20, 30],
        'clf__max_features': ['sqrt', 'log2'],
        'clf__min_samples_split': sp_randint(2, 11),
        'clf__min_samples_leaf': sp_randint(1, 6),
        'clf__bootstrap': [True, False]
    },
    "GradientBoosting": {
        'clf__n_estimators': sp_randint(100, 500),
        'clf__learning_rate': sp_uniform(0.01, 0.29),
        'clf__max_depth': sp_randint(2, 6),
        'clf__subsample': [0.6, 0.8, 1.0],
        'clf__min_samples_split': sp_randint(2, 11),
        'clf__min_samples_leaf': sp_randint(1, 6)
    }
}

# ---------------------------------------------------------------------
# 4️⃣  Train and tune each model
# ---------------------------------------------------------------------
summary, models = {}, {}
start = time.time()

# Logistic Regression
print("\n🔹 Tuning Logistic Regression ...")
pipe_log = Pipeline([('pre', preprocessor),
                     ('clf', LogisticRegression(max_iter=1000, random_state=42))])
gs_log = GridSearchCV(pipe_log, grids["Logistic"], cv=3, scoring='accuracy', n_jobs=-1)
gs_log.fit(X_train, y_train)
y_pred = gs_log.predict(X_test)
summary["Logistic"] = {
    "params": gs_log.best_params_,
    "cv_score": gs_log.best_score_,
    "test_acc": accuracy_score(y_test, y_pred)
}
models["Logistic"] = gs_log.best_estimator_

# Random Forest
print("\n🌲 Tuning Random Forest ...")
pipe_rf = Pipeline([('pre', preprocessor),
                    ('clf', RandomForestClassifier(random_state=42))])
rs_rf = RandomizedSearchCV(pipe_rf, grids["RandomForest"], n_iter=80, cv=3,
                           scoring='accuracy', n_jobs=-1, random_state=42)
rs_rf.fit(X_train, y_train)
y_pred = rs_rf.predict(X_test)
summary["RandomForest"] = {
    "params": rs_rf.best_params_,
    "cv_score": rs_rf.best_score_,
    "test_acc": accuracy_score(y_test, y_pred)
}
models["RandomForest"] = rs_rf.best_estimator_

# Gradient Boosting
print("\n⚡ Tuning Gradient Boosting ...")
pipe_gb = Pipeline([('pre', preprocessor),
                    ('clf', GradientBoostingClassifier(random_state=42))])
rs_gb = RandomizedSearchCV(pipe_gb, grids["GradientBoosting"], n_iter=100, cv=3,
                           scoring='accuracy', n_jobs=-1, random_state=42)
rs_gb.fit(X_train, y_train)
y_pred = rs_gb.predict(X_test)
summary["GradientBoosting"] = {
    "params": rs_gb.best_params_,
    "cv_score": rs_gb.best_score_,
    "test_acc": accuracy_score(y_test, y_pred)
}
models["GradientBoosting"] = rs_gb.best_estimator_

# ---------------------------------------------------------------------
# 5️⃣  Select best model & evaluate
# ---------------------------------------------------------------------
best_name = max(summary, key=lambda m: summary[m]['test_acc'])
best_model = models[best_name]
print(f"\n✅ Best model: {best_name}")

cv5 = cross_val_score(best_model, X, y_enc, cv=5, scoring='accuracy')
summary["Overall"] = {
    "best_model": best_name,
    "cv5_mean": cv5.mean(),
    "cv5_std": cv5.std(),
    "elapsed_min": (time.time()-start)/60
}

# ---------------------------------------------------------------------
# 6️⃣  Save artifacts
# ---------------------------------------------------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, f"models/final_best_{best_name}_pipeline.pkl")
joblib.dump(le, "models/label_encoder.pkl")
pd.DataFrame(summary).to_csv("models/training_summary.csv")

print("\n🎯 Training completed.")
print("Summary:\n", summary)
print("\nArtifacts saved in ./models/")
