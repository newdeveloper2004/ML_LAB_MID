import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "input/data.csv"


def load_data():
    return pd.read_csv(DATA_PATH)


def get_reg_split():
    """Regression: predict petal_length (a number) from the other measurements."""
    df = load_data()
    X = df[["sepal_length", "sepal_width", "petal_width"]]
    y = df["petal_length"]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def get_clf_split():
    """Classification: predict species (a category) from the 4 measurements."""
    df = load_data()
    X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    y = df["species"]
    return train_test_split(X, y, test_size=0.2, random_state=42)
