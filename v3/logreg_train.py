import sys
import os
import pandas as pd
import math


def load_dataset() -> pd.DataFrame:
    assert len(sys.argv) == 2, "usage: logreg_train.py <dataset/path>"

    path = sys.argv[1]
    assert os.path.isfile(path), "invalid dataset path"

    df = pd.read_csv(path)
    return df


def min_max_normalization(value, v_min, v_max) -> float:
    return (value - v_min) / (v_max - v_min)


def polynomial_regression(x: float, thetas: list[float]) -> float:
    return 0


def sigmoid_hypothesis_function(x: float, thetas: list[float]) -> float:
    return 1 / (1 + math.exp(- polynomial_regression(x, thetas)))


def log_loss(x, y, t0, t1) -> float:
    if y == 1:
        return - math.log(sigmoid_hypothesis_function(x, t0, t1))
    else:
        return - math.log(1 - sigmoid_hypothesis_function(x, t1, t0))


def gradient_descent(t1, t0) -> tuple[float, float]:
    pass


def logistic_regression(X: pd.DataFrame, y: pd.DataFrame):
    # normaliser la donnée
    # calculer la fonction sigmoid hypothese en utilisant une regression lineaire multiple
    # calculer le cout (log loss) de model actuel
    # répéter une nombre fini de fois l'algo: Epoch
    # utiliser la technique du one vs all pour appliquer la regression selon les 4 maisons
    pass


def main():
    try:
        df = load_dataset()
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)
        df.drop(inplace=True, columns="Index")
        X = df.drop(columns=["Hogwarts House"])
        y = df["Hogwarts House"]
        # peut etre utiliser 'get_dummies' sur les features discretes/string pour beneficier de plus de data
        logistic_regression(X, y)
    except Exception as e:
        print(f"{e.__class__.__name__}: {e.args[0]}")


if __name__ == "__main__":
    main()
