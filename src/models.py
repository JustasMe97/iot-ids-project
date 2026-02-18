from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

# Random Forest modelis
# n_estimators – medžių skaičius
def get_random_forest(n_estimators=150, random_state=42):
    return RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )

# SVM modelis (dabar nenaudoju)
def get_svm(random_state=42):
    # probability=True reikalinga ROC/AUC (predict_proba)
    return SVC(kernel="rbf", probability=True, random_state=random_state)

# Paprastas MLP be skalavimo
def get_mlp(random_state=42):
    return MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        max_iter=30,    #del greicio, siaip reikia daugiau
        random_state=random_state
    )

# MLP su duomenų standartizavimu
def get_mlp_scaled(random_state=42):
    # Pipeline leidžia sujungti skalavimą ir modelį į vieną objektą
    return Pipeline(steps=[
        ("scaler", StandardScaler(with_mean=False)),  # with_mean=False saugiau su sparse/one-hot
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            max_iter=150,             # max iteraciju skaicius
            early_stopping=True,      # sustos anksčiau jei nebegerėja
            n_iter_no_change=10,      # kiek iteracijų laukti pagerėjimo
            random_state=random_state
        ))
    ])