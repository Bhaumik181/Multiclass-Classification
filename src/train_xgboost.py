import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import os

PREP_PATH = "../data/preprocessed/"
MODEL_PATH = "../model/"
os.makedirs(MODEL_PATH, exist_ok=True)

# -----------------------
# Load data
# -----------------------
X = np.load(PREP_PATH + "X.npy")
y = np.load(PREP_PATH + "y.npy")

# XGBoost needs classes starting from 0
# Your original classes are 5–11, so shift them to 0–6
y = y - 5

FOLDS = 4
EPOCHS = 1000

skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

models = []
scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f"\n================ XGBoost Fold {fold} ================")

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model = XGBClassifier(
        n_estimators=EPOCHS,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=7,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Predict on validation set
    val_probs = model.predict_proba(X_val)
    val_preds = np.argmax(val_probs, axis=1)

    acc = accuracy_score(y_val, val_preds)
    print(f"Fold {fold} Accuracy: {acc * 100:.2f}%")

    models.append(model)
    scores.append(acc)

print("\n====================================")
print(f"Average XGBoost CV Accuracy: {np.mean(scores) * 100:.2f}%")
print("====================================")

# Save all trained models
joblib.dump(models, MODEL_PATH + "xgb_models.pkl")
print("XGBoost models saved as: model/xgb_models.pkl")
