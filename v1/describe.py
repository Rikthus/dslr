import sys
import os
import pandas as pd


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


def std(values: pd.Series) -> float:
    assert len(values) != 0, "<values> should not be empty"
    return var(values)**0.5


def max(values: pd.Series) -> float:
    assert len(values) != 0, "<values> should not be empty"
    v_max = values[0]
    for val in values:
        if val > v_max:
            v_max = val
    return v_max


def min(values: pd.Series) -> float:
    assert len(values) != 0, "<values> should not be empty"
    v_min = values[0]
    for val in values:
        if val < v_min:
            v_min = val
    return v_min


def median(values: pd.Series) -> float:
    assert len(values) != 0, "<values> should not be empty"
    m = len(values)
    values_lst = values.to_list()
    if m % 2 != 0:
        return values_lst[m // 2]
    else:
        med = (values_lst[m // 2 - 1] + values_lst[m // 2]) / 2
        return med


def quartiles(values: pd.Series) -> float:
    assert len(values) != 0, "<values> should not be empty"
    m = len(values)

    q1 = median(values[:m // 2])
    q3 = median(values[m // 2:])
    return (q1, q3)


def describe(data: pd.DataFrame):
    describe_dict = {
        "Feature": [],
        "Count": [],
        "Mean": [],
        "Std": [],
        "Min": [],
        "25%": [],
        "50%": [],
        "75%": [],
        "Max": []
    }
    if isinstance(data, pd.DataFrame):
        for key, values in data.items():
            if len(values) > 0 and isinstance(values[0], (float | int)):
                q1, q3 = quartiles(values)
                describe_dict["Feature"].append(key)
                describe_dict["Count"].append(round(float(len(values)), 6))
                describe_dict["Mean"].append(round(mean(values), 6))
                describe_dict["Std"].append(round(std(values), 6))
                describe_dict["Min"].append(round(min(values), 6))
                describe_dict["25%"].append(round(q1, 6))
                describe_dict["50%"].append(round(median(values), 6))
                describe_dict["75%"].append(round(q3, 6))
                describe_dict["Max"].append(round(max(values), 6))

        for key, stat in describe_dict.items():
            print(f"{key:<10}", *stat, sep='         ')


def main():
    try:
        df = load_dataset()
        df.drop(columns=["Index"], axis=1, inplace=True)
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)
        describe(df)
    except Exception as e:
        print(f"{e.__class__.__name__}: {e.args[0]}")


if __name__ == "__main__":
    main()
