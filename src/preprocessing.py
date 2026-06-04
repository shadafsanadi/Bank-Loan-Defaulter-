"""
Preprocessing pipeline.

Builds a sklearn ColumnTransformer that handles:
  - Numeric columns: median imputation + standard scaling
  - Categorical columns: mode imputation + ordinal encoding

Saving the fitted preprocessor as a sklearn object (not separate functions)
guarantees identical transformations at train time and inference time,
eliminating train/serve skew.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from src.utils.logger import get_logger

logger = get_logger(__name__)

HIGH_MISSING_THRESHOLD = 0.40
OUTLIER_CAP_PERCENTILE = 99


def remove_high_missing_columns(df: pd.DataFrame, threshold: float = HIGH_MISSING_THRESHOLD) -> pd.DataFrame:
    """Drop columns where the fraction of missing values exceeds threshold."""
    missing_frac = df.isnull().mean()
    drop_cols = missing_frac[missing_frac > threshold].index.tolist()
    if drop_cols:
        logger.info(f"Dropping {len(drop_cols)} high-missing columns: {drop_cols[:5]}...")
    return df.drop(columns=drop_cols, errors="ignore")


def remove_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with invalid CODE_GENDER=XNA (data quality artifact)."""
    if "CODE_GENDER" in df.columns:
        mask = df["CODE_GENDER"] == "XNA"
        if mask.sum() > 0:
            logger.info(f"Removing {mask.sum()} rows with CODE_GENDER=XNA")
        return df[~mask].reset_index(drop=True)
    return df


def cap_outliers(df: pd.DataFrame, percentile: int = OUTLIER_CAP_PERCENTILE) -> pd.DataFrame:
    """Cap extreme outliers in skewed financial columns at the given percentile."""
    skewed_cols = [
        "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
        "BUREAU_DEBT_SUM", "BUREAU_CREDIT_SUM", "BUREAU_OVERDUE_SUM",
    ]
    for col in skewed_cols:
        if col in df.columns:
            cap = float(df[col].quantile(percentile / 100))
            n_capped = (df[col] > cap).sum()
            if n_capped > 0:
                df[col] = df[col].astype(float).clip(upper=cap)
    return df


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build and return a sklearn ColumnTransformer for the given DataFrame.

    The transformer is NOT fit here — call .fit_transform(X_train) to fit,
    then .transform(X_test) for evaluation and inference.

    Categorical columns use OrdinalEncoder rather than OneHotEncoder because:
    1. One-hot encoding 58-category ORGANIZATION_TYPE creates 57 sparse columns
    2. XGBoost/LightGBM handle ordinal-encoded categoricals natively
    3. Feature space stays manageable (~300 features instead of ~500+)
    """
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "bool"]).columns.tolist()

    logger.info(f"Preprocessor: {len(num_cols)} numeric, {len(cat_cols)} categorical columns")

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_cols),
            ("cat", categorical_pipeline, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


# ── Backward-compatible functions (used by legacy src/train.py and app.py) ────

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Legacy function — prefer build_preprocessor() for new code."""
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    for col in cat_cols:
        mode_vals = df[col].mode()
        if not mode_vals.empty:
            df[col] = df[col].fillna(mode_vals.iloc[0])
    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Legacy function — prefer build_preprocessor() for new code."""
    return pd.get_dummies(df, drop_first=True)
