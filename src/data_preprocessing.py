import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
import joblib, os

RAW_PATH = "../data/raw/"
PREP_PATH = "../data/preprocessed/"
MODEL_PATH = "../model/"

os.makedirs(PREP_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)

# -----------------------
# Load Data
# -----------------------
train = pd.read_csv(RAW_PATH + "train.csv")
test = pd.read_csv(RAW_PATH + "test.csv")

X = train.drop("target", axis=1)
y = train["target"].values
X_test = test.drop("id", axis=1)

print("Data loaded")

# -----------------------
# 1. Missing Value Handling
# -----------------------
imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(X)
X_test = imputer.transform(X_test)

# -----------------------
# 2. Remove Near-Constant Features
# -----------------------
var_filter = VarianceThreshold(threshold=0.001)
X = var_filter.fit_transform(X)
X_test = var_filter.transform(X_test)

print("After variance filter:", X.shape)

# -----------------------
# 3. Robust Scaling (light, not aggressive)
# -----------------------
scaler = RobustScaler(quantile_range=(5, 95))
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)

# -----------------------
# 4. Feature Selection using Mutual Information
# This is the BIG accuracy booster
# -----------------------
k = 700   # keep best 700 features, tune this (600–800)
selector = SelectKBest(score_func=mutual_info_classif, k=k)

X = selector.fit_transform(X, y)
X_test = selector.transform(X_test)

print("After mutual information selection:", X.shape)

# -----------------------
# Save Everything
# -----------------------
np.save(PREP_PATH + "X.npy", X)
np.save(PREP_PATH + "y.npy", y)
np.save(PREP_PATH + "X_test.npy", X_test)

joblib.dump(imputer, MODEL_PATH + "imputer.pkl")
joblib.dump(var_filter, MODEL_PATH + "variance_filter.pkl")
joblib.dump(scaler, MODEL_PATH + "scaler.pkl")
joblib.dump(selector, MODEL_PATH + "feature_selector.pkl")

print("Optimized preprocessing completed successfully.")
