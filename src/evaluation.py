from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
# Modelio įvertinimas naudojant numatytąjį threshold (0.5)
def evaluate_model(model, X_test, y_test, print_reports=True):
    # Modelio klasės prognozės
    y_pred = model.predict(X_test) 
    #Tikimybės, kad įrašas priklauso "atakos" klasei
    y_probs = model.predict_proba(X_test)[:, 1] 
    #confusion matrix isskaidymas i TN, FP, FN, TP
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel() 

    # Pagrindiniu metriku apskaiciavimas
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_probs),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    }

    if print_reports: # Jei reikia detalios ataskaitos spausdinimas
        print("\nClassification report:")
        print(classification_report(y_test, y_pred))
        print("\nConfusion matrix:")
        print(confusion_matrix(y_test, y_pred))

     # ROC kreivės duomenys
    fpr, tpr, _ = roc_curve(y_test, y_probs)

    return {
        "metrics": metrics,
        "fpr": fpr,
        "tpr": tpr
    }

# Įvertinimas naudojant rankiniu būdu pasirinktą threshold
def evaluate_from_probs(y_test, y_probs, threshold=0.5):
    y_pred = (y_probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auc": roc_auc_score(y_test, y_probs),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp
    }

    # ROC duomenys
    fpr, tpr, _ = roc_curve(y_test, y_probs)

    return {"metrics": metrics, "fpr": fpr, "tpr": tpr}