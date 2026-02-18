import pandas as pd

# Metodas atskirai paruoštų train ir test rinkinių įkėlimui
def load_dataset(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

# Įkelia vieną CSV failą, testuojant kitus rinkinius.
def load_single_dataset(path):
    return pd.read_csv(path)