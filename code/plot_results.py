import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

models = ["kNN","SVM","XGBoost","ANN","Decision Tree","Random Forest"]
accuracy = [95.45,94.12,96.78,99.35,98.50,99.95]
precision = [0.953,0.941,0.967,0.993,0.985,0.9995]
recall = [0.955,0.944,0.968,0.994,0.986,0.9997]
f1 = [0.954,0.943,0.967,0.993,0.985,0.9996]

def bar_plot(values, title, ylabel, filename):
    plt.figure()
    plt.bar(models, values)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(f"plots/{filename}", dpi=300)
    plt.close()

bar_plot(accuracy, "Accuracy Comparison", "Accuracy (%)", "fig1_accuracy.png")
bar_plot(precision, "Precision Comparison", "Precision", "fig2_precision.png")
bar_plot(recall, "Recall Comparison", "Recall", "fig3_recall.png")
bar_plot(f1, "F1-score Comparison", "F1-score", "fig4_f1.png")

# Confusion Matrix
cm = np.array([[9987,5],[3,10005]])
plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normal","Attack"],
            yticklabels=["Normal","Attack"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix – Random Forest")
plt.savefig("plots/fig6_cm.png", dpi=300)
plt.close()
