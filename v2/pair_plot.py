import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def load_dataset() -> pd.DataFrame:
    assert len(sys.argv) == 2, "usage: describe.py <dataset/path>"

    path = sys.argv[1]
    assert os.path.isfile(path), "invalid dataset path"

    df = pd.read_csv(path)
    return df


def identify_non_numeric_columns(df: pd.DataFrame) -> list[str]:
    non_numeric_columns = []
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            non_numeric_columns.append(col)
    return non_numeric_columns


def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(scaled_data, columns=df.columns)
    return df_scaled


# COLOR BY HOUSE
def plot_most_similar_features(df: pd.DataFrame):
    sns.pairplot(df, hue="House", palette="colorblind")
    plt.tight_layout()
    plt.show()


def main():
    try:
        df = load_dataset()
        df.drop(columns=["Index"], axis=1, inplace=True)
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)
        df['House'] = pd.Categorical(df['Hogwarts House']).codes
        non_numeric_columns = identify_non_numeric_columns(df)
        non_numeric_columns.remove("Hogwarts House")
        filtered_df = df.drop(columns=non_numeric_columns, axis=1)
        filtered_df_scaled = normalize_data(filtered_df.drop(["Hogwarts House"], axis=1))
        plot_most_similar_features(filtered_df_scaled)
    except Exception as e:
        print(f"{e.__class__.__name__}: {e.args[0]}")


if __name__ == "__main__":
    main()
