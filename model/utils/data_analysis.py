import numpy as np
import pandas as pd
from model.utils.chi_square import perform_chi_square_test
from scipy.stats import shapiro


def get_uncorrelated_col(df, threshold):
    numeric_df = df.select_dtypes(exclude=["object"])
    x = numeric_df.drop(["SalePrice"], axis=1)
    y = numeric_df["SalePrice"]
    corr_y = []
    for i in x.columns:
        corr_val = np.corrcoef(x[i], y)
        if abs(corr_val[0][1]) < threshold:
            corr_y.append((i, round(float(corr_val[0][1]), 3)))
    return corr_y


def get_insignificant_columns(df):
    categorical = df.select_dtypes(include=["object"]).columns
    col_significance = [(col, perform_chi_square_test(df, col, "SalePrice")) for col in categorical]
    col_to_drop = []
    for sig in col_significance:
        if sig[1][2] == False:
            col_to_drop.append((sig[0], round(float(sig[1][1]), 3)))
    return col_to_drop


def get_missing_values(df):
    missing_values_cols = []
    missing_values_count = df.isnull().sum()
    missing_values_count = missing_values_count + df.isna().sum()
    numeric_df = df.select_dtypes(exclude=["object"])
    for col in missing_values_count[missing_values_count > 0].index:
        if col in numeric_df.columns:
            missing_values_cols.append((col, bool(shapiro(numeric_df[col].dropna())[1] > 0.05)))
        else:
            missing_values_cols.append((col, False))
    return missing_values_cols


def get_data_for_preprocessing(df, treshhold):
    uncorrelated_col = get_uncorrelated_col(df, treshhold)
    insignificant_col = get_insignificant_columns(df)
    missing_values_col = get_missing_values(df)
    return [x[0] for x in uncorrelated_col], [y[0] for y in insignificant_col], missing_values_col
