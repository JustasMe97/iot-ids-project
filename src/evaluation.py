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

def evaluate_multiclass(y_test, y_pred, labels=None, target_names=None, print_reports=True):
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    report = classification_report(
        y_test, y_pred,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_precision": report["weighted avg"]["precision"],
        "weighted_recall": report["weighted avg"]["recall"],
        "weighted_f1": report["weighted avg"]["f1-score"],
    }

    per_class = {}
    for k, v in report.items():
        if k in ("accuracy", "macro avg", "weighted avg"):
            continue
        per_class[k] = {
            "precision": v["precision"],
            "recall": v["recall"],
            "f1": v["f1-score"],
            "support": int(v["support"]),
        }

    worst_classes_by_recall = sorted(
        [(cls, d["recall"], d["support"]) for cls, d in per_class.items()],
        key=lambda x: (x[1], x[2])
    )

    if print_reports:
        print("\nClassification report (multiclass):")
        print(classification_report(
            y_test, y_pred,
            labels=labels,
            target_names=target_names,
            zero_division=0
        ))
        print("\nConfusion matrix:")
        print(cm)

        if len(worst_classes_by_recall) > 0:
            print("\nBlogiausiai aptinkamos atakos (by recall):")
            for cls, rec, sup in worst_classes_by_recall[:10]:
                print(f"{cls:20s} recall={rec:.3f}  support={sup}")

    return {
        "metrics": metrics,
        "cm": cm,
        "per_class": per_class,
        "worst_classes_by_recall": worst_classes_by_recall
    }