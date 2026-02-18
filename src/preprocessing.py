import pandas as pd

# Pašalina nereikalingus stulpelius
# "attack_cat" nenaudojam dvejetainei klasifikacijai
# "id" neturi informacinės vertės modeliui
def clean_dataframe(df):
    return df.drop(columns=["id", "attack_cat"], errors="ignore")

# Atskiriam požymius (X) ir tikslinę reikšmę (y)
def split_features_target(df, target="label"):
    X = df.drop(columns=[target]) # visi stulpeliai išskyrus label
    y = df[target]                # tikslinė klasė (0 arba 1)
    return X, y

# One-hot kodavimas kategoriniams požymiams
def encode_data(X_train, X_test=None):
    # Kategoriniai kintamieji paverčiami į skaitinius (0/1 stulpeliai)
    X_train = pd.get_dummies(X_train)

    # Sulyginame test stulpelius su train stulpeliais
    # Jei test rinkinyje nėra kokios kategorijos užpildom 0
    if X_test is not None:
        X_test = pd.get_dummies(X_test)
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
        return X_train, X_test

    return X_train