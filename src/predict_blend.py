import numpy as np
import pandas as pd
import joblib
import os

RAW_PATH = "../data/raw/"
PREP_PATH = "../data/preprocessed/"
MODEL_PATH = "../model/"
SUB_PATH = "../submissions/"
ANALYSIS_PATH = "../submissions/analysis/"

os.makedirs(SUB_PATH, exist_ok=True)
os.makedirs(ANALYSIS_PATH, exist_ok=True)

# -----------------------
# Load Data
# -----------------------
print("Loading test data and models...")
X_test = np.load(PREP_PATH + "X_test.npy")
models = joblib.load(MODEL_PATH + "advanced_lgbm_models.pkl")

test = pd.read_csv(RAW_PATH + "test.csv")
test_ids = test["id"]

num_classes = 7   # classes: 5–11
num_models = len(models)

print(f"{num_models} models loaded.")

# -----------------------
# Probability Ensemble
# -----------------------
print("Running ensemble predictions...")
probs = np.zeros((X_test.shape[0], num_classes))

for i, model in enumerate(models, 1):
    model_probs = model.predict_proba(X_test)
    probs += model_probs
    print(f"Model {i} contributed.")

# Average probabilities
probs /= num_models

# -----------------------
# (Optional) Temperature Scaling
# Makes probabilities smoother and better calibrated
# -----------------------
USE_TEMPERATURE = True
TEMPERATURE = 1.2   # >1 = softer probabilities, <1 = sharper

if USE_TEMPERATURE:
    print("Applying temperature scaling...")
    probs = np.exp(np.log(probs + 1e-12) / TEMPERATURE)
    probs = probs / probs.sum(axis=1, keepdims=True)

# -----------------------
# Final Predictions
# -----------------------
final_preds = np.argmax(probs, axis=1) + 5
confidence = np.max(probs, axis=1)

# -----------------------
# Confidence-aware adjustment (optional)
# If confidence is extremely low, we can trust majority class
# -----------------------
CONF_THRESHOLD = 0.40

majority_class = np.bincount(final_preds).argmax()

low_conf_mask = confidence < CONF_THRESHOLD
final_preds[low_conf_mask] = majority_class

# -----------------------
# Save submission
# -----------------------
submission = pd.DataFrame({
    "id": test_ids,
    "target": final_preds
})

submission.to_csv(SUB_PATH + "submission.csv", index=False)
print("Final submission saved at submissions/submission.csv")

# -----------------------
# Save analysis file
# -----------------------
analysis_df = pd.DataFrame(probs, columns=[f"prob_{i+5}" for i in range(num_classes)])
analysis_df["id"] = test_ids
analysis_df["prediction"] = final_preds
analysis_df["confidence"] = confidence

analysis_df.to_csv(ANALYSIS_PATH + "prediction_analysis.csv", index=False)

print("Prediction analysis file saved at submissions/analysis/prediction_analysis.csv")

print("\nSample predictions:")
print(submission.head())

print("\nSample confidence:")
print(analysis_df[["id", "prediction", "confidence"]].head())
