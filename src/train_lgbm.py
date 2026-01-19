import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import joblib, os

PREP_PATH = "../data/preprocessed/"
MODEL_PATH = "../model/"
os.makedirs(MODEL_PATH, exist_ok=True)

# Load preprocessed data
X = np.load(PREP_PATH + "X.npy")
y = np.load(PREP_PATH + "y.npy")

FOLDS = 4
EPOCHS = 3000   # epochs = maximum number of trees

skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

models = []
scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f"\n================ Fold {fold} ================")

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model = lgb.LGBMClassifier(
        n_estimators=EPOCHS,
        learning_rate=0.03,
        num_leaves=128,
        max_depth=-1,
        min_data_in_leaf=30,
        min_gain_to_split=0.001,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.2,
        objective="multiclass",
        num_class=7,
        random_state=42,
        verbosity=-1
    )

    # Callbacks for early stopping (compatible with all LightGBM versions)
    callbacks = [
        lgb.early_stopping(stopping_rounds=100, verbose=False),
        lgb.log_evaluation(period=0)
    ]

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="multi_logloss",
        callbacks=callbacks
    )

    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    print(f"Fold {fold} Accuracy: {acc * 100:.2f}%")

    models.append(model)
    scores.append(acc)

print("\n====================================")
print(f"Average CV Accuracy: {np.mean(scores) * 100:.2f}%")
print("====================================")

# IMPORTANT: Save with the exact name your predict/evaluate expects
joblib.dump(models, MODEL_PATH + "advanced_lgbm_models.pkl")

print("Advanced LightGBM models saved as: model/advanced_lgbm_models.pkl")
