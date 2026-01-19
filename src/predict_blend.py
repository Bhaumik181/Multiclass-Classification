import numpy as np
import pandas as pd
import joblib
import os

# -----------------------
# Paths
# -----------------------
RAW_PATH = "../data/raw/"
PREP_PATH = "../data/preprocessed/"
MODEL_PATH = "../model/"
SUB_PATH = "../submissions/"

os.makedirs(SUB_PATH, exist_ok=True)

# -----------------------
# Auto-increment filename
# -----------------------
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

# -----------------------
# Load Test Data
# -----------------------
print("Loading test data...")

# For CatBoost (DataFrame – keeps feature names)
test_df = pd.read_csv(RAW_PATH + "test.csv")
test_ids = test_df["id"]
X_test_df = test_df.drop("id", axis=1)

# For LightGBM / XGBoost (NumPy – if available)
if os.path.exists(PREP_PATH + "X_test.npy"):
    X_test_arr = np.load(PREP_PATH + "X_test.npy")
else:
    print("Warning: X_test.npy not found, using raw values.")
    X_test_arr = X_test_df.values

# -----------------------
# Load Models
# -----------------------
print("Loading models and label encoder...")

# MUST exist
cat_models = joblib.load(MODEL_PATH + "cat_models.pkl")
label_encoder = joblib.load(MODEL_PATH + "label_encoder.pkl")

# Optional (if you trained them)
try:
    lgbm_models = joblib.load(MODEL_PATH + "advanced_lgbm_models.pkl")
    xgb_models = joblib.load(MODEL_PATH + "xgb_models.pkl")
    has_others = True
    print("LightGBM and XGBoost models loaded.")
except:
    print("LightGBM/XGBoost not found. Running CatBoost only.")
    has_others = False
    lgbm_models = []
    xgb_models = []

num_classes = len(label_encoder.classes_)

# -----------------------
# Ensemble helper
# -----------------------
def ensemble(models, X, name):
    probs = np.zeros((X.shape[0], num_classes))
    for i, model in enumerate(models, 1):
        p = model.predict_proba(X)
        probs += p
        print(f"{name} Model {i} done.")
    return probs / len(models)

# -----------------------
# Run Ensembles
# -----------------------
print("\nRunning CatBoost ensemble...")
cat_probs = ensemble(cat_models, X_test_df, "CatBoost")

lgbm_probs = None
xgb_probs = None

if has_others:
    print("\nRunning LightGBM ensemble...")
    lgbm_probs = ensemble(lgbm_models, X_test_arr, "LightGBM")

    print("\nRunning XGBoost ensemble...")
    xgb_probs = ensemble(xgb_models, X_test_arr, "XGBoost")

# -----------------------
# Blending Strategy
# CatBoost is strongest
# -----------------------
if has_others:
    final_probs = (
        0.00 * lgbm_probs +   # small diversity
        1.0 * cat_probs  +   # main brain
        0.00 * xgb_probs      # secondary strong model
    )
else:
    final_probs = cat_probs

# -----------------------
# Final Predictions
# -----------------------
pred_indices = np.argmax(final_probs, axis=1)

# Decode: 0–6 → 5–11
final_predictions = label_encoder.inverse_transform(pred_indices)
confidence = np.max(final_probs, axis=1)

# -----------------------
# Save Submission
# -----------------------
submission = pd.DataFrame({
    "id": test_ids,
    "target": final_predictions
})

submission_file = get_next_submission_filename(SUB_PATH, "submission")
submission.to_csv(submission_file, index=False)

print(f"\nSubmission file created: {submission_file}")
print(submission.head())

# -----------------------
# Save Analysis (Optional but Useful)
# -----------------------
analysis = pd.DataFrame(
    final_probs,
    columns=[f"prob_{cls}" for cls in label_encoder.classes_]
)
analysis["id"] = test_ids
analysis["prediction"] = final_predictions
analysis["confidence"] = confidence

analysis_file = get_next_submission_filename(SUB_PATH, "prediction_analysis")
analysis.to_csv(analysis_file, index=False)

print(f"\nPrediction analysis saved: {analysis_file}")
print(analysis[["id", "prediction", "confidence"]].head())
