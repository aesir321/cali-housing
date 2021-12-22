import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split


def label_strata(dataset: pd.DataFrame) -> pd.DataFrame:
    dataset["income_cat"] = pd.cut(
        dataset["median_income"],
        bins=[0, 1.5, 3, 4.5, 6, np.inf],
        labels=[1, 2, 3, 4, 5],
    )
    dataset["house_val_cat"] = pd.cut(
        dataset["median_house_value"],
        bins=[0, 150000, 200000, 250000, 300000, 350000, np.inf],
        labels=[1, 2, 3, 4, 5, 6],
    )

    return dataset


def create_train_test_sets(
    dataset: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    strata_dataset = label_strata(dataset)

    X = dataset.drop("median_house_value", axis=1)
    y = dataset["median_house_value"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=strata_dataset[["house_val_cat", "income_cat"]]
    )
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    dataset = pd.read_csv("housing.csv")
    X_train, X_test, y_train, y_test = create_train_test_sets(dataset)
