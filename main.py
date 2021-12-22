import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVR, SVR
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error

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
        X, y, stratify=strata_dataset[["house_val_cat", "income_cat"]]
    )
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    dataset = pd.read_csv("housing.csv")
    X_train, X_test, y_train, y_test = create_train_test_sets(dataset)

    num_features = [
        "longitude",
        "latitude",
        "housing_median_age",
        "total_rooms",
        "total_bedrooms",
        "population",
        "households",
        "median_income",
    ]
    num_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("median_imputer", SimpleImputer(strategy="median")),
            ("feature_creator", FeatureCreator()),
        ]
    )

    cat_features = ["ocean_proximity"]
    cat_pipeline = Pipeline([("onehot", OneHotEncoder())])

    pipeline = ColumnTransformer(
        [("num", num_pipeline, num_features), ("cat", cat_pipeline, cat_features)]
    )

    X_train_t = pipeline.fit_transform(X_train, y_train)

    linear_svm = LinearSVR(random_state=42)
    linear_svm.fit(X_train_t, y_train)

    y_pred = linear_svm.predict(X_train_t)
    rmse = mean_squared_error(y_train, y_pred, squared=False)
    print(f"RMSE: ${rmse}")
