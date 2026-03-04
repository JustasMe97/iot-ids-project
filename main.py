from sklearn.preprocessing import LabelEncoder
from src.data_loader import load_dataset
from src.preprocessing import clean_dataframe, split_features_target, encode_data
from src.models import get_mlp_scaled, get_random_forest, get_svm, get_mlp
from src.evaluation import evaluate_from_probs, evaluate_model
from src.evaluation import evaluate_multiclass
from src.visualization import plot_roc_multi, plot_feature_importance
import time
import pandas as pd

print("Programa startavo")
# paruostu UNSW train/test rinkiniu ikelimas
train_df, test_df = load_dataset(
    "data/UNSW_NB15_training-set(in).csv",
    "data/UNSW_NB15_testing-set(in).csv"
)

TASK = "multiclass"  # "binary" arba "multiclass"

# --- duomenu paruosimo laiko(preprocessing time) skaiciavimas ---
start_prep = time.time()

# Pašalinam nereikalingu stulpelius (id ir attack_cat ar tik id, priklausomai nuo binary ar multiclass rezimo)
train_df = clean_dataframe(train_df, task=TASK)
test_df = clean_dataframe(test_df, task=TASK)

# Atskiriam požymius (X) ir target (y)
if TASK == "binary":
    X_train, y_train = split_features_target(train_df, target="label")
    X_test, y_test = split_features_target(test_df, target="label")
else:
    X_train, y_train = split_features_target(train_df, target="attack_cat")
    X_test, y_test = split_features_target(test_df, target="attack_cat")

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)
    class_names = le.classes_

# One-hot kodavimas kategoriniams požymiams + stulpelių sulyginimas
X_train, X_test = encode_data(X_train, X_test)

end_prep = time.time()
preprocessing_time = end_prep - start_prep
print(f"\nPreprocessing time: {preprocessing_time:.4f} sec")

#Paruošiam modelių sąrašą testavimui
rf_estimators_list = [30, 150] # pasirenkam modeliui medžių skaičų RF

models = {}

for n in rf_estimators_list:
    models[f"RF_{n}"] = get_random_forest(n_estimators=n)

# Pridedam MLP
models["MLP"] = get_mlp_scaled()

roc_items = []
results_table = []

#Paleidžiam mokymą ir vertinimą kiekvienam modeliui
for name, model in models.items():
    print(f"\n=== {name} ===")

    # --- modelio mokymo laiko(training time) skaiciavimas ---
    start_train = time.time()
    model.fit(X_train, y_train)
    end_train = time.time()
    training_time = end_train - start_train

    if TASK == "multiclass":
        if "RF" in name:
            plot_feature_importance(model, X_train.columns, top_n=15)

    # Modelio prognozavimo trukmė (inference time)
    start_inf = time.time()

    if TASK == "binary":
        #y_pred = model.predict(X_test) # jeigu darom su default treshold
        threshold = 0.5   # treshold
        y_probs = model.predict_proba(X_test)[:, 1] # Tikimybės, kad įrašas yra ataka (label=1)
        y_pred = (y_probs >= threshold).astype(int) # Klasė pagal pasirinktą slenkstį
    else:
        y_pred = model.predict(X_test) # multiclass: prognozuojam klases tiesiogiai

    end_inf = time.time()
    inference_time = end_inf - start_inf
    time_per_sample = inference_time / len(X_test)

    #Vertinam metrikas
    if TASK == "binary":
        res = evaluate_from_probs(y_test, y_probs, threshold=threshold)
        metrics = res["metrics"]

        # Išsisaugo ROC kreivės duomenis bendram grafiko piešimui
        roc_items.append((name, res["fpr"], res["tpr"], metrics["auc"]))

        # Išsisaugo metrikas palyginimo lentelei
        results_table.append({
            "Model": name,
            "Accuracy": metrics["accuracy"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1": metrics["f1"],
            "AUC": metrics["auc"],
            "TP": metrics["tp"],
            "FP": metrics["fp"],
            "FN": metrics["fn"],
            "TN": metrics["tn"],
            "Training_time_sec": training_time,
            "Inference_time_sec": inference_time,
            "Time_per_sample_ms": time_per_sample * 1000
        })

    else:
        res = evaluate_multiclass(
            y_test, y_pred,
            labels=list(range(len(class_names))),
            target_names=class_names,
            print_reports=True
        )
        metrics = res["metrics"]

        # Išsisaugo metrikas palyginimo lentelei
        results_table.append({
            "Model": name,
            "Accuracy": metrics["accuracy"],
            "Macro_F1": metrics["macro_f1"],
            "Weighted_F1": metrics["weighted_f1"],
            "Macro_Precision": metrics["macro_precision"],
            "Macro_Recall": metrics["macro_recall"],
            "Training_time_sec": training_time,
            "Inference_time_sec": inference_time,
            "Time_per_sample_ms": time_per_sample * 1000
        })

if TASK == "binary":
    # --------- ROC grafikas ----
    plot_roc_multi(roc_items)

# --------- LENTELĖ ---------
df_results = pd.DataFrame(results_table)

if TASK == "binary":
    df_results = df_results.sort_values(by="AUC", ascending=False)
else:
    df_results = df_results.sort_values(by="Weighted_F1", ascending=False)

print("\nModelių palyginimo lentelė:")
print(df_results.to_string(index=False))