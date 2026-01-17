import numpy as np
import pandas as pd
import joblib
import os

RAW_PATH = "../data/raw/"
PREP_PATH = "../data/preprocessed/"
MODEL_PATH = "../model/"
SUB_PATH = "../submissions/"

os.makedirs(SUB_PATH, exist_ok=True)

print("Loading test data...")
X_test = np.load(PREP_PATH + "X_test.npy")

print("Loading trained models...")
models = joblib.load(MODEL_PATH + "advanced_lgbm_models.pkl")

test = pd.read_csv(RAW_PATH + "test.csv")
test_ids = test["id"]

num_classes = 7  # target classes: 5 to 11
num_models = len(models)

print(f"{num_models} models loaded. Starting ensemble prediction...")

# ---------------------------------
# Probability ensemble
# ---------------------------------
ensemble_probs = np.zeros((X_test.shape[0], num_classes))

for i, model in enumerate(models, 1):
    probs = model.predict_proba(X_test)
    ensemble_probs += probs
    print(f"Model {i} done.")

ensemble_probs /= num_models

# ---------------------------------
# Temperature scaling (optional)
# ---------------------------------
USE_TEMPERATURE = True
TEMPERATURE = 1.2

if USE_TEMPERATURE:
    print("Applying temperature scaling...")
    ensemble_probs = np.exp(np.log(ensemble_probs + 1e-12) / TEMPERATURE)
    ensemble_probs = ensemble_probs / ensemble_probs.sum(axis=1, keepdims=True)

# ---------------------------------
# Final predictions
# ---------------------------------
final_predictions = np.argmax(ensemble_probs, axis=1) + 5
confidence = np.max(ensemble_probs, axis=1)

# ---------------------------------
# Save submission
# ---------------------------------
submission = pd.DataFrame({
    "id": test_ids,
    "target": final_predictions
})

submission.to_csv(SUB_PATH + "submission.csv", index=False)

print("\nSubmission file created: submissions/submission.csv")
print(submission.head())

# ---------------------------------
# Optional: Save confidence + probabilities
# ---------------------------------
analysis = pd.DataFrame(
    ensemble_probs,
    columns=[f"prob_{i+5}" for i in range(num_classes)]
)
analysis["id"] = test_ids
analysis["prediction"] = final_predictions
analysis["confidence"] = confidence

analysis.to_csv(SUB_PATH + "prediction_analysis.csv", index=False)

print("\nPrediction analysis saved: submissions/prediction_analysis.csv")
print(analysis[["id", "prediction", "confidence"]].head())
