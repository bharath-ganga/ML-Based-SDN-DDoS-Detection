import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import random

# ---------------- REPRODUCIBILITY ----------------
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ---------------- LOAD DATA ----------------
df = pd.read_csv("data/CICIDS2017_SYN.csv")

# ---------------- LABEL ENCODING ----------------
# Normal = 0, Attack = 1
df["Label"] = df["Label"].map({"Normal": 0, "Attack": 1})

# ---------------- FEATURE SELECTION ----------------
features = [
    "Flow Duration",
    "Total Fwd Packets",
    "Packet Length Variance",
    "Average Packet Size",
    "ACK Flag Count",
    "Idle Mean",
    "SYN Count"
]

X = df[features]
y = df["Label"]

# ---------------- NORMALIZATION ----------------
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# ---------------- TRAIN–TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=20000,        # EXACT paper condition
    random_state=SEED,
    stratify=y
)

# ---------------- RANDOM FOREST MODEL ----------------
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    bootstrap=True,
    random_state=SEED,
    n_jobs=-1
)

rf.fit(X_train, y_train)

# ---------------- SAVE MODEL ----------------
joblib.dump(rf, "models/rf_model.pkl")

print("Random Forest trained and saved successfully.")
