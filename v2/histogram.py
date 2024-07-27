import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_dataset() -> pd.DataFrame:
    assert len(sys.argv) == 2, "usage: describe.py <dataset/path>"

    path = sys.argv[1]
    assert os.path.isfile(path), "invalid dataset path"

    df = pd.read_csv(path)
    return df


def mean(values: pd.Series) -> float:
    assert len(values) != 0, "<values> should not be empty"
    m = len(values)
    return values.sum() / m


def var(values: pd.Series) -> float:
    assert len(values) != 0, "<values> should not be empty"
    m = len(values)
    v_mean = mean(values)
    squared_mean_dist = []
    for x in values:
        squared_mean_dist.append((x - v_mean)**2)
    df_squared_mean_dist = pd.Series(squared_mean_dist)
    return df_squared_mean_dist.sum() / m


def identify_non_numeric_columns(df: pd.DataFrame) -> list[str]:
    non_numeric_columns = []
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            non_numeric_columns.append(col)
    return non_numeric_columns


def get_variances_by_house(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    variances_by_house = {
        "Ravenclaw": {},
        "Slytherin": {},
        "Gryffindor": {},
        "Hufflepuff": {},
    }
    groups = df.groupby("Hogwarts House")
    for name, group in groups:
        group.drop(["Hogwarts House"], axis=1, inplace=True)
        for class_name, values in group.items():
            variances_by_house[name][class_name] = var(values)
    return variances_by_house


def draw_plots(var_df: dict[str, pd.DataFrame], ft_names: list[str]):
    fig, axes = plt.subplots(4, 4, figsize=(17, 10))

    for index, class_name in enumerate(ft_names):
        x = index // 4
        y = index % 4

        sns.barplot(x='House', y='Value', data=var_df[class_name], ax=axes[x, y])
        axes[x, y].set_xlabel('House')
        axes[x, y].set_ylabel('Variance')
        axes[x, y].set_title(class_name)

    plt.tight_layout()
    plt.show()


def plot_homogeneity(df: pd.DataFrame):
    variances_by_house = get_variances_by_house(df)
    df.drop(["Hogwarts House"], axis=1, inplace=True)

    var_df = {}
    for class_name in df.columns:
        variances = []
        for values in variances_by_house.values():
            variances.append(values[class_name])
        data = {col: [val] for col, val in zip(variances_by_house.keys(), variances)}
        mini_df = pd.DataFrame(data)
        df_arranged = mini_df.melt(var_name='House', value_name='Value')

        var_df[class_name] = df_arranged
    draw_plots(var_df, df.columns)


def main():
    try:
        df = load_dataset()
        df.drop(columns=["Index"], axis=1, inplace=True)
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)
        non_numeric_columns = identify_non_numeric_columns(df)
        non_numeric_columns.remove("Hogwarts House")
        filtered_df = df.drop(columns=non_numeric_columns, axis=1)
        plot_homogeneity(filtered_df)

    except Exception as e:
        print(f"{e.__class__.__name__}: {e.args[0]}")


if __name__ == "__main__":
    main()
