# main.py
import pickle, json, numpy as np, pandas as pd
from pathlib import Path

MODEL_DIR = "models"
MODEL_PATHS = {
    "hypertension": MODEL_DIR+"/hypertension_predict_model",
    "stroke":       MODEL_DIR+"/stroke_predict_model",
    "diabetes":     MODEL_DIR+"/diabetes_predict_model",
    "obesity":      MODEL_DIR+"/obesity_predict_model",
    "heart":        MODEL_DIR+"/heart_predict_model",
}


MODEL_INPUT_KEYS = {
    "hypertension": ["gender", "age", "currentSmoker", "cigsPerDay", "diabetes",
                     "sysBP", "diaBP", "BMI", "heartRate",
                     "chol_category", "glucose_category"],
    "stroke": ["gender", "age", "hypertension", "heart_disease", "ever_married",
               "work_type", "Residence_type", "BMI", "smoking_status"],
    "diabetes": ["Pregnancies", "BloodPressure", "SkinThickness", "BMI",
                 "DiabetesPedigreeFunction", "age", "glucose_category"],
    "obesity": ["sex", "age", "Height", "Weight"],
    "heart": ["age", "sex", "trestbps", "fbs", "thalach", "exang",
              "chol_category"],
}

BINARY_MAP = {"yes": 1, "no": 0, "male": 1, "female": 0}
CATEGORY_MAP = {
    "chol_category":      {"low_intake": 0, "moderate_intake": 1,
                           "high_intake": 2},
    "glucose_category":   {"low_intake": 0, "moderate_intake": 1,
                           "high_intake": 2, "very_high_intake": 3},
    "smoking_status":     {"never_smoked": 0, "formerly_smoked": 1,
                           "smokes": 2},
    "work_type":          {"child": 0, "never_worked": 1, "self_employed": 2,
                           "govt_job": 3, "private": 4},
    "Residence_type":     {"urban": 1, "rural": 0},
    "DiabetesPedigreeFunction":
        {"no_history": 0.0, "low_history": 0.25,
         "moderate_history": 0.5, "high_history": 0.75},
    "exang": BINARY_MAP,
}

GLUCOSE_ALIASES = {
    "low": "low_intake", "moderate": "moderate_intake",
    "high": "high_intake", "very_high": "very_high_intake",
}

print("Please enter the following details:")
ui = {
    "gender": input("Gender (male/female): ").lower(),
    "age": int(input("Age: ")),
    "Height": float(input("Height (m): ")),
    "Weight": float(input("Weight (kg): ")),
    "sysBP": float(input("Systolic BP: ")),
    "diaBP": float(input("Diastolic BP: ")),
    "heartRate": int(input("Heart Rate (bpm): ")),
    "cigsPerDay": int(input("Cigarettes Per Day: ")),
    "smoking_status": input("Smoking Status (never_smoked/formerly_smoked/smokes): ").lower(),
    "chol_category": input("Cholesterol Intake (low_intake/moderate_intake/high_intake): ").lower(),
    "glucose_category": input("Glucose Intake (low/moderate/high/very_high): ").lower(),
    "ever_married": input("Ever Married (yes/no): ").lower(),
    "Pregnancies": int(input("Number of Pregnancies: ")),
    "work_type": input("Work Type (child/never_worked/self_employed/govt_job/private): ").lower(),
    "Residence_type": input("Residence Type (urban/rural): ").lower(),
    "DiabetesPedigreeFunction": input("Family History of Diabetes (no_history/low_history/moderate_history/high_history): ").lower(),
    "exang": input("Chest Pain During Exercise (yes/no): ").lower(),
}


if ui["glucose_category"] in GLUCOSE_ALIASES:
    ui["glucose_category"] = GLUCOSE_ALIASES[ui["glucose_category"]]

ui["BMI"] = round(ui["Weight"] / (ui["Height"] ** 2), 2)
ui["currentSmoker"] = 1 if ui["smoking_status"] == "smokes" else 0
ui["sex"] = BINARY_MAP.get(ui["gender"], 0)

glucose_code = CATEGORY_MAP["glucose_category"].get(ui["glucose_category"], 0)
ui["fbs"] = 1 if glucose_code > 2 else 0

ui["trestbps"] = ui["sysBP"]
ui["thalach"]  = ui["heartRate"]


for key in ["diabetes", "hypertension", "heart_disease",
            "BloodPressure", "SkinThickness"]:
    ui.setdefault(key, 0)


preds = {}
comp_probs = []

for name, path in MODEL_PATHS.items():
    with open(path, "rb") as f:
        model = pickle.load(f)

    keys = MODEL_INPUT_KEYS[name]
    vec = []

    for k in keys:
        v = ui.get(k, 0)
        # mapping
        if k in CATEGORY_MAP:
            v = CATEGORY_MAP[k].get(v, 0)
        elif isinstance(v, str) and v in BINARY_MAP:
            v = BINARY_MAP[v]
        vec.append(float(v))

    proba = (model.predict_proba([vec])[0][1]
             if hasattr(model, "predict_proba")
             else model.predict([vec])[0])

    proba = max(0.05, min(0.95, proba))
    comp_probs.append(proba)

    percent = round(proba * 100, 2)
    if name == "obesity":
        if ui["BMI"] < 18.5:
            category = "Underweight"
        elif ui["BMI"] < 24.9:
            category = "Normal"
        elif ui["BMI"] < 29.9:
            category = "Overweight"
        else:
            category = "Obese"
        preds[f"{name}_risk_percent"] = {
            "percentage": percent, "category": category
        }
    else:
        label = (f"Risk of having {name}"
                 if percent > 50 else f"Risk of not having {name}")
        preds[f"{name}_risk_percent"] = {
            "percentage": percent, "belongingness": label
        }

meta_prob = float(np.mean(comp_probs))
mets_flag = 1 if meta_prob >= 0.5 else 0

preds["metabolic_syndrome"] = {
    "probability_percent": round(meta_prob * 100, 2),
    "MetS_flag": int(mets_flag)
}


with open("health_predictions.json", "w") as f:
    json.dump(preds, f, indent=4)

print("\nAll predictions (including MetS) saved to health_predictions.json")
print(f"Metabolic Syndrome flag: {mets_flag}  "
      f"({round(meta_prob*100,2)} % probability)")
