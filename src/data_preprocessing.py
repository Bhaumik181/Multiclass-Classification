import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Paths
RAW_PATH = "../data/raw/"
PREP_PATH = "../data/preprocessed/"
MODEL_PATH = "../model/"

os.makedirs(PREP_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)

# Load data
train = pd.read_csv(RAW_PATH + "train.csv")
test = pd.read_csv(RAW_PATH + "test.csv")

# Separate features and target
X = train.drop("target", axis=1)
y = train["target"]

# Split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
test_scaled = scaler.transform(test.drop("id", axis=1))

# Save numpy arrays
np.save(PREP_PATH + "X_train.npy", X_train_scaled)
np.save(PREP_PATH + "X_val.npy", X_val_scaled)
np.save(PREP_PATH + "y_train.npy", y_train.to_numpy())
np.save(PREP_PATH + "y_val.npy", y_val.to_numpy())
np.save(PREP_PATH + "test_scaled.npy", test_scaled)

# Save scaler
joblib.dump(scaler, MODEL_PATH + "scaler.pkl")

print("Data preprocessing completed and saved.")
