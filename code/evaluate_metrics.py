import numpy as np

# CONFUSION MATRIX VALUES (FROM PAPER)
TN = 9987
FP = 5
FN = 3
TP = 10005

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
