import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVR, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import mlflow.sklearn
from sklearn.datasets import fetch_california_housing
from scipy.stats import reciprocal, uniform

from FeatureCreator import FeatureCreator


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
        X, y, stratify=strata_dataset[["house_val_cat", "income_cat"]], random_state=42
    )
    return X_train, X_test, y_train, y_test


def create_pipeline(X: np.ndarray) -> np.ndarray:
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("median_imputer", SimpleImputer(strategy="median")),
            ("feature_creator", FeatureCreator()),
        ]
    )

    X_train_t = pipeline.fit_transform(X_train)

    return X_train_t


if __name__ == "__main__":
    housing = fetch_california_housing()
    X = housing["data"]
    y = housing["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    X_train_t = create_pipeline(X_train)

    mlflow.autolog()
    param_distributions = {"gamma": reciprocal(0.001, 0.1), "C": uniform(1, 10)}
    rnd_search_cv = RandomizedSearchCV(
        SVR(),
        param_distributions,
        n_iter=10,
        verbose=2,
        cv=3,
        random_state=42,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    with mlflow.start_run() as run:
        rnd_search_cv.fit(X_train_t, y_train)
