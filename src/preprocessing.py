import pandas as pd

# Pašalina nereikalingus stulpelius
# "attack_cat" nenaudojam dvejetainei klasifikacijai(tik multiclass)
# "id" neturi informacinės vertės modeliui
def clean_dataframe(df: pd.DataFrame, task: str = "binary") -> pd.DataFrame:
    df = df.copy()

    # 'id' trynimas
    drop_cols = ["id"]

    if task == "binary":
        drop_cols += ["attack_cat"]   # panaikinam attack_cat binary versijai
    elif task == "multiclass":
        drop_cols += ["label"]        # panaikinam label muticlass versijai
    else:
        raise ValueError("task must be 'binary' or 'multiclass'")

    return df.drop(columns=drop_cols, errors="ignore")

# Atskiriam požymius (X) ir tikslinę reikšmę (y)
def split_features_target(df: pd.DataFrame, target: str):
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataframe columns.")

    X = df.drop(columns=[target])
    y = df[target]
    return X, y

# One-hot kodavimas kategoriniams požymiams
def encode_data(X_train, X_test=None):
    # Kategoriniai kintamieji paverčiami į skaitinius (0/1 stulpeliai)
    X_train_enc = pd.get_dummies(X_train)

    # Sulyginame test stulpelius su train stulpeliais
    # Jei test rinkinyje nėra kokios kategorijos užpildom 0
    if X_test is None:
        return X_train_enc

    X_test_enc = pd.get_dummies(X_test)
    X_test_enc = X_test_enc.reindex(columns=X_train_enc.columns, fill_value=0)
    return X_train_enc, X_test_enc