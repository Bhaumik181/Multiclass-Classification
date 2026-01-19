import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import os

# -----------------------
# Paths
# -----------------------
RAW_PATH = "../data/raw/"
MODEL_PATH = "../model/"
os.makedirs(MODEL_PATH, exist_ok=True)

# -----------------------
# Load Data
# -----------------------
print("Loading training data...")
train_df = pd.read_csv(RAW_PATH + "train.csv")

X = train_df.drop("target", axis=1)
y = train_df["target"]

# Encode labels: 5–11 → 0–6
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print("Target mapping:")
for i, cls in enumerate(le.classes_):
    print(f"{cls} → {i}")

# -----------------------
# Training Configuration
# -----------------------
FOLDS = 5
EPOCHS = 1200
LEARNING_RATE = 0.03

skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

models = []
scores = []

# -----------------------
# Training Loop
# -----------------------
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_encoded), 1):
    print(f"\n================ CatBoost Fold {fold} ================")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]

    model = CatBoostClassifier(
        iterations=EPOCHS,
        learning_rate=LEARNING_RATE,
        depth=8,
        loss_function="MultiClass",
        eval_metric="Accuracy",

        # Regularization
        l2_leaf_reg=5,
        random_strength=1,

        # Use Bernoulli bootstrap to allow subsampling
        bootstrap_type="Bernoulli",
        subsample=0.85,
        colsample_bylevel=0.7,

        # CPU optimization
        thread_count=-1,

        # Early stopping
        early_stopping_rounds=100,

        random_seed=42,
        verbose=200
    )

    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        use_best_model=True
    )

    preds = model.predict(X_val).reshape(-1)
    acc = accuracy_score(y_val, preds)
    print(f"Fold {fold} Accuracy: {acc * 100:.2f}%")

    models.append(model)
    scores.append(acc)

# -----------------------
# Results
# -----------------------
print("\n====================================")
print(f"Average CatBoost CV Accuracy: {np.mean(scores) * 100:.2f}%")
print("====================================")

# -----------------------
# Save Models and Encoder
# -----------------------
joblib.dump(models, MODEL_PATH + "cat_models.pkl")
joblib.dump(le, MODEL_PATH + "label_encoder.pkl")

print("CatBoost models saved at: model/cat_models.pkl")
print("Label encoder saved at: model/label_encoder.pkl")
