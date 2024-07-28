import json
import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


WEIGHTS_FILE = "weights.json"
CLASSES_TO_REMOVE = [
    "Care of Magical Creatures",
    "Arithmancy",
    "Astronomy"
]


def load_dataset() -> pd.DataFrame:
    assert len(sys.argv) == 2, "usage: logreg_train.py <dataset/path>"

    path = sys.argv[1]
    assert os.path.isfile(path), "invalid dataset path"

    df = pd.read_csv(path)
    return df


def remove_useless_features(df: pd.DataFrame) -> pd.DataFrame:
    df.drop(columns=CLASSES_TO_REMOVE, inplace=True)
    return df


def get_weights() -> dict[str, list]:
    assert os.path.isfile(WEIGHTS_FILE), f"no {WEIGHTS_FILE} found, you should train the model first"

    with open(WEIGHTS_FILE, 'r') as f:
        weights = json.load(f)
    return weights


def identify_non_numeric_columns(df: pd.DataFrame) -> list[str]:
    non_numeric_columns = []
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            non_numeric_columns.append(col)
    return non_numeric_columns


def standardize_data(df: pd.DataFrame) -> pd.DataFrame:
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(scaled_data, columns=df.columns)
    return df_scaled


def format_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df.drop(inplace=True, columns="Index")
    non_numeric_columns = identify_non_numeric_columns(df)
    x_df = standardize_data(df.drop(columns=non_numeric_columns, axis=1))
    return x_df


def sigmoid_function(X: np.ndarray, thetas: np.ndarray) -> np.ndarray:
    z = X.dot(thetas)
    val = 1 / (1 + np.exp(-z))
    return val


def predict(df: pd.DataFrame, weights: dict[str, list]):
    pass


def main():
    try:
        weights = get_weights()
        df = load_dataset()
        filtered_df = remove_useless_features(df)
        x_df = format_dataset(filtered_df)
        predict(x_df, weights)
        # save_predictions_in_file()
    except Exception as e:
        print(f"{e.__class__.__name__}: {e.args[0]}")


if __name__ == "__main__":
    main()
