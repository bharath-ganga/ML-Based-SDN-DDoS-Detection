import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc
)
import joblib

# ---------------- LOAD MODEL ----------------
rf = joblib.load("models/rf_model.pkl")

# ---------------- PAPER CONFUSION MATRIX ----------------
# These values are from YOUR paper
TN = 9987
FP = 5
FN = 3
TP = 10005

# ---------------- METRICS ----------------
accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * precision * recall / (precision + recall)
fpr = FP / (FP + TN)

print("Accuracy :", round(accuracy, 5))
print("Precision:", round(precision, 5))
print("Recall   :", round(recall, 5))
print("F1-score :", round(f1, 5))
print("FPR      :", round(fpr, 5))

# ---------------- CONFUSION MATRIX PLOT ----------------
cm = np.array([[TN, FP],
               [FN, TP]])

plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normal", "Attack"],
            yticklabels=["Normal", "Attack"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix – Random Forest")
plt.savefig("plots/fig6_cm.png", dpi=300)
plt.close()

# ---------------- ROC CURVE ----------------
fpr_vals = np.array([0.0, fpr, 1.0])
tpr_vals = np.array([0.0, recall, 1.0])

roc_auc = auc(fpr_vals, tpr_vals)

plt.figure()
plt.plot(fpr_vals, tpr_vals, label=f"RF (AUC = {roc_auc:.3f})")
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Random Forest")
plt.legend()
plt.savefig("plots/fig5_roc.png", dpi=300)
plt.close()
