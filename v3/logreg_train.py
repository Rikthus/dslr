import sys
import os
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


WEIGHTS_FILE = "weights.json"
LEARNING_RATE = 0.1
EPOCHS = 1000

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


def save_weights_in_file(weights: dict[str, list]):
    with open(WEIGHTS_FILE, 'w') as f:
        json.dump(weights, f)


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


def format_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove Nan and duplicates in the original dataset. Then separate it in 2 datasets,
    one without non numeric features and one containing a column for each House.
    Inside these columns there is a 1 if the student is from this House, else a 0.

    The numeric features values are standardized.

    Parameters:
        df : pandas.DataFrame
            Dataset to format

    Returns:
        tuple[pandas.DataFrame, pandas.DataFrame]:
            Dataset of standardized numeric features only and Dataset of Students memberships
    """
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df.drop(inplace=True, columns="Index")
    non_numeric_columns = identify_non_numeric_columns(df)

    x_df = standardize_data(df.drop(columns=non_numeric_columns, axis=1))
    y_df = pd.get_dummies(df["Hogwarts House"])
    for col in y_df:
        y_df[col] = y_df[col].replace({True: 1, False: 0})

    return (x_df, y_df)


def remove_useless_features(df: pd.DataFrame) -> pd.DataFrame:
    df.drop(columns=CLASSES_TO_REMOVE, inplace=True)
    return df


def gradient_descent(X: np.ndarray, h: np.ndarray, Y: np.ndarray, thetas: np.ndarray) -> np.ndarray:
    m = Y.shape[0]

    gradients = X.T.dot((h - Y)) / m
    thetas = thetas - LEARNING_RATE * gradients

    return thetas


def sigmoid_function(X: np.ndarray, thetas: np.ndarray) -> np.ndarray:
    z = X.dot(thetas)
    val = 1 / (1 + np.exp(-z))
    return val


def logistic_regression(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    thetas = np.random.randn(X.shape[1], 1)

    for _ in range(EPOCHS):
        h = sigmoid_function(X, thetas)
        thetas = gradient_descent(X, h, Y, thetas)

    return thetas


def one_vs_all(x_df: pd.DataFrame, y_df: pd.DataFrame):
    x = x_df.to_numpy()
    X = np.hstack((x, np.ones((x.shape[0], 1))))

    weights = {}
    for col in y_df:
        y = y_df[col].to_numpy()
        Y = y.reshape(y.shape[0], 1).astype(float)

        thetas = logistic_regression(X, Y)
        weights[col] = thetas.tolist()
    return weights


def main():
    try:
        pd.set_option('future.no_silent_downcasting', True)
        df = load_dataset()
        filtered_df = remove_useless_features(df)
        x_df, y_df = format_dataset(filtered_df)
        weights = one_vs_all(x_df, y_df)
        save_weights_in_file(weights)
    except Exception as e:
        print(f"{e.__class__.__name__}: {e.args[0]}")


if __name__ == "__main__":
    main()
