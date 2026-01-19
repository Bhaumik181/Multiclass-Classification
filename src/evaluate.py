import numpy as np
import pandas as pd
import joblib
import os

RAW_PATH = "../data/raw/"
PREP_PATH = "../data/preprocessed/"
MODEL_PATH = "../model/"
SUB_PATH = "../submissions/"

os.makedirs(SUB_PATH, exist_ok=True)

# --------------------------------------------------
# Auto-increment submission filenames
# --------------------------------------------------
def get_next_submission_filename(folder, base_name="submission", ext=".csv"):
    existing = [f for f in os.listdir(folder) if f.startswith(base_name) and f.endswith(ext)]
    nums = []
    for f in existing:
        try:
            nums.append(int(f.replace(base_name, "").replace(ext, "").replace("_", "")))
        except:
            pass
    next_num = max(nums) + 1 if nums else 1
    return os.path.join(folder, f"{base_name}_{next_num}{ext}")

# --------------------------------------------------
# Load Data
# --------------------------------------------------
print("Loading test data...")
X_test = np.load(PREP_PATH + "X_test.npy")

test = pd.read_csv(RAW_PATH + "test.csv")
test_ids = test["id"]

print("Loading models...")
lgbm_models = joblib.load(MODEL_PATH + "advanced_lgbm_models.pkl")
cat_models  = joblib.load(MODEL_PATH + "cat_models.pkl")
xgb_models  = joblib.load(MODEL_PATH + "xgb_models.pkl")

num_classes = 7   # classes: 5–11

# --------------------------------------------------
# Ensemble helper
# --------------------------------------------------
def ensemble(models, X, name):
    probs = np.zeros((X.shape[0], num_classes))
    for i, model in enumerate(models, 1):
        probs += model.predict_proba(X)
        print(f"{name} Model {i} done.")
    return probs / len(models)

# --------------------------------------------------
# Run individual ensembles
# --------------------------------------------------
print("\nRunning LightGBM ensemble...")
lgbm_probs = ensemble(lgbm_models, X_test, "LightGBM")

print("\nRunning CatBoost ensemble...")
cat_probs = ensemble(cat_models, X_test, "CatBoost")

print("\nRunning XGBoost ensemble...")
xgb_probs = ensemble(xgb_models, X_test, "XGBoost")

# --------------------------------------------------
# Final weighted blend
# CatBoost dominates since it is your strongest model
# --------------------------------------------------
final_probs = (
    0.25 * lgbm_probs +
    0.55 * cat_probs +
    0.20 * xgb_probs
)

# --------------------------------------------------
# Temperature scaling (optional but useful)
# --------------------------------------------------
USE_TEMPERATURE = True
TEMPERATURE = 1.2

if USE_TEMPERATURE:
    print("Applying temperature scaling...")
    final_probs = np.exp(np.log(final_probs + 1e-12) / TEMPERATURE)
    final_probs = final_probs / final_probs.sum(axis=1, keepdims=True)

# --------------------------------------------------
# Final predictions
# --------------------------------------------------
final_predictions = np.argmax(final_probs, axis=1) + 5
confidence = np.max(final_probs, axis=1)

# --------------------------------------------------
# Save submission with new incremental name
# --------------------------------------------------
submission = pd.DataFrame({
    "id": test_ids,
    "target": final_predictions
})

submission_file = get_next_submission_filename(SUB_PATH, "submission")
submission.to_csv(submission_file, index=False)

print(f"\nSubmission file created: {submission_file}")
print(submission.head())

# --------------------------------------------------
# Save analysis file (optional but very useful)
# --------------------------------------------------
analysis = pd.DataFrame(
    final_probs,
    columns=[f"prob_{i+5}" for i in range(num_classes)]
)
analysis["id"] = test_ids
analysis["prediction"] = final_predictions
analysis["confidence"] = confidence

analysis_file = get_next_submission_filename(SUB_PATH, "prediction_analysis")
analysis.to_csv(analysis_file, index=False)

print(f"\nPrediction analysis saved: {analysis_file}")
print(analysis[["id", "prediction", "confidence"]].head())
