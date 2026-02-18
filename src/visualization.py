import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import numpy as np

# ROC kreives keliems modeliams viename grafike piesimas
# roc_items: sąrašas (modelio_pavadinimas, fpr, tpr, auc)
def plot_roc_multi(roc_items):
    """
    roc_items: list of tuples (name, fpr, tpr, auc)
    """
    plt.figure()
    # Kiekvienam modeliui nubraižom po ROC kreivę
    for name, fpr, tpr, auc in roc_items:
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

    # Įstrižainė – atsitiktinio spėjimo (random guess) linija
    plt.plot([0, 1], [0, 1])

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (model comparison)")
    plt.legend()
    plt.show()

# ROC kreivė vienam modeliui
def plot_roc(y_test, y_probs):
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1])  # atsitiktinio spėjimo linija
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.show()

# Požymių svarbos grafikas
def plot_feature_importance(model, feature_names, top_n=10):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1] # surikiuoja nuo svarbiausio

    plt.figure()
    plt.barh(
        [feature_names[i] for i in indices[:top_n]],
        importances[indices[:top_n]]
    )
    plt.gca().invert_yaxis() # kad svarbiausias būtų viršuje
    plt.title("Top Feature Importances")
    plt.show()