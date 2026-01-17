import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, PowerTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
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
# 2. Remove Low Variance Features
# -----------------------
var_filter = VarianceThreshold(threshold=0.0001)
X = var_filter.fit_transform(X)
X_test = var_filter.transform(X_test)

print("After variance filter:", X.shape)

# -----------------------
# 3. Robust Scaling (better for outliers)
# -----------------------
scaler = RobustScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)

# -----------------------
# 4. Power Transformation
# Makes data more Gaussian-like
# -----------------------
power = PowerTransformer(method="yeo-johnson")
X = power.fit_transform(X)
X_test = power.transform(X_test)

# -----------------------
# 5. Correlation Filtering
# Remove highly correlated features
# -----------------------
X_df = pd.DataFrame(X)
corr_matrix = X_df.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
X = X_df.drop(columns=to_drop).values
X_test = pd.DataFrame(X_test).drop(columns=to_drop).values

print("After correlation filter:", X.shape)

# -----------------------
# 6. (Optional) PCA
# Set USE_PCA = True if you want dimensionality reduction
# -----------------------
USE_PCA = False   # change to True if you want PCA

if USE_PCA:
    pca = PCA(n_components=0.95, random_state=42)
    X = pca.fit_transform(X)
    X_test = pca.transform(X_test)
    joblib.dump(pca, MODEL_PATH + "pca.pkl")
    print("After PCA:", X.shape)

# -----------------------
# Save Everything
# -----------------------
np.save(PREP_PATH + "X.npy", X)
np.save(PREP_PATH + "y.npy", y)
np.save(PREP_PATH + "X_test.npy", X_test)

joblib.dump(imputer, MODEL_PATH + "imputer.pkl")
joblib.dump(var_filter, MODEL_PATH + "variance_filter.pkl")
joblib.dump(scaler, MODEL_PATH + "scaler.pkl")
joblib.dump(power, MODEL_PATH + "power_transformer.pkl")

print("Advanced preprocessing completed successfully.")
