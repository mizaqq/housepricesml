import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler, Normalizer
from utils.data_analysis import get_data_for_preprocessing
from sklearn.model_selection import train_test_split
import logging
from typing import List, Sequence


def drop_columns(df: pd.DataFrame, insignificant_col: Sequence[str], uncorrelated_col: Sequence[str]) -> pd.DataFrame:
    df = df.drop(columns=insignificant_col)
    df = df.drop(columns=uncorrelated_col)
    return df


def fill_missing_values(df: pd.DataFrame, missing_values_col: Sequence[str]) -> pd.DataFrame:
    numerical = df.select_dtypes(exclude=["object"])
    categorical = df.select_dtypes(include=["object"])
    for col in missing_values_col:
        if col[1] == False and (col[0] in numerical.columns):
            df[col[0]] = df[col[0]].fillna(df[col[0]].median())
        elif col[1] == True and (col[0] in numerical.columns):
            df[col[0]] = df[col[0]].fillna(df[col[0]].mean())
        elif col[0] in categorical.columns:
            df[col[0]] = df[col[0]].fillna(df[col[0]].mode()[0])
    return df


def encode_numeric(df: pd.DataFrame) -> pd.DataFrame:
    categorical_numeric = [
        "MSZoning",
        "Street",
        "LotShape",
        "LotConfig",
        "ExterQual",
        "ExterCond",
        "Foundation",
        "BsmtQual",
        "BsmtCond",
        "BsmtExposure",
        "Heating",
        "KitchenQual",
        "FireplaceQu",
        "GarageFinish",
        "GarageQual",
        "SaleCondition",
    ]
    le = LabelEncoder()
    for col in categorical_numeric:
        df[col] = le.fit_transform(df[col])
    return df


def encode_onehot(df: pd.DataFrame) -> pd.DataFrame:
    categorical_onehot = ["Neighborhood", "CentralAir", "SaleType"]

    for col in categorical_onehot:
        onehot = pd.get_dummies(df[col], prefix=col, dummy_na=True)
        df.drop(col, axis=1, inplace=True)
        df = df.join(onehot)

    return df


def preprocess_data(df: pd.DataFrame, df_test=None) -> pd.DataFrame:
    uncorrelated_col, insignificant_col, missing_values_col = get_data_for_preprocessing(df, treshhold=0.05)
    logging.info(f"Uncorrelated columns: {uncorrelated_col}")
    logging.info(f"Insignificant columns: {insignificant_col}")
    logging.info(f"Missing values columns: {missing_values_col}")
    df = drop_columns(df, insignificant_col, uncorrelated_col)
    df = fill_missing_values(df, missing_values_col)
    df = encode_numeric(df)
    df = encode_onehot(df)
    if df_test is not None:
        df_test = drop_columns(df_test, insignificant_col, uncorrelated_col)
        df_test = fill_missing_values(df_test, missing_values_col)
        df_test = encode_numeric(df_test)
        df_test = encode_onehot(df_test)
        return df, df_test
    return df


def scale_data(df: pd.DataFrame) -> pd.DataFrame:
    scaler = StandardScaler()
    df = scaler.fit_transform(df)
    return df


def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    normalized_df = (df - df.mean()) / df.std()
    return normalized_df


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 0) -> tuple:
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop("SalePrice", axis=1), df["SalePrice"], test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test
