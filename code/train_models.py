import pandas as pd
import numpy as np
import random
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

# ---------------- REPRODUCIBILITY ----------------
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ---------------- LOAD DATA ----------------
df = pd.read_csv("data/CICIDS2017_SYN.csv")

df["Label"] = df["Label"].map({"Normal": 0, "Attack": 1})

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
X = scaler.fit_transform(X)

# ---------------- SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=20000, stratify=y, random_state=SEED
)

# ---------------- MODELS ----------------
models = {
    "kNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(kernel="rbf", probability=True),
    "DecisionTree": DecisionTreeClassifier(random_state=SEED),
    "ANN": MLPClassifier(hidden_layer_sizes=(64,32), max_iter=300, random_state=SEED),
    "RandomForest": RandomForestClassifier(
        n_estimators=200, random_state=SEED, n_jobs=-1
    )
}

# ---------------- TRAIN & SAVE ----------------
for name, model in models.items():
    model.fit(X_train, y_train)
    joblib.dump(model, f"models/{name}.pkl")

print("All models trained and saved.")
