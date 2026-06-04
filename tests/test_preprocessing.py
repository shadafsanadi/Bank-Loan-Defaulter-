"""Tests for preprocessing pipeline."""

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from src.preprocessing import (
    remove_high_missing_columns,
    remove_invalid_rows,
    build_preprocessor,
    cap_outliers,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "AMT_INCOME_TOTAL": [100_000, 50_000, 200_000, 1_000_000_000],
        "AMT_CREDIT":       [500_000, 300_000, 800_000, 200_000],
        "EXT_SOURCE_2":     [0.7, 0.3, 0.9, 0.1],
        "CODE_GENDER":      ["F", "M", "XNA", "F"],
        "NAME_INCOME_TYPE": ["Working", "Pensioner", "Working", "Unemployed"],
        "HIGH_MISSING_COL": [np.nan, np.nan, np.nan, 1.0],  # 75% missing
    })


class TestRemoveHighMissingColumns:

    def test_drops_column_above_threshold(self, sample_df):
        result = remove_high_missing_columns(sample_df, threshold=0.4)
        assert "HIGH_MISSING_COL" not in result.columns

    def test_keeps_columns_below_threshold(self, sample_df):
        result = remove_high_missing_columns(sample_df, threshold=0.4)
        assert "AMT_INCOME_TOTAL" in result.columns
        assert "EXT_SOURCE_2" in result.columns

    def test_custom_threshold(self, sample_df):
        result = remove_high_missing_columns(sample_df, threshold=0.9)
        assert "HIGH_MISSING_COL" in result.columns  # only 75% missing, below 90%


class TestRemoveInvalidRows:

    def test_removes_xna_rows(self, sample_df):
        result = remove_invalid_rows(sample_df)
        assert "XNA" not in result["CODE_GENDER"].values

    def test_preserves_valid_rows(self, sample_df):
        result = remove_invalid_rows(sample_df)
        assert len(result) == 3  # 1 XNA row removed

    def test_handles_missing_code_gender_column(self):
        df = pd.DataFrame({"A": [1, 2, 3]})
        result = remove_invalid_rows(df)
        assert len(result) == 3  # no-op


class TestCapOutliers:

    def test_caps_extreme_income(self, sample_df):
        result = cap_outliers(sample_df.copy(), percentile=99)
        assert result["AMT_INCOME_TOTAL"].max() < 1_000_000_000

    def test_does_not_affect_normal_values(self, sample_df):
        original_median = sample_df["AMT_INCOME_TOTAL"].median()
        result = cap_outliers(sample_df.copy(), percentile=99)
        # Median should be unchanged
        assert result["AMT_INCOME_TOTAL"].median() == original_median


class TestBuildPreprocessor:

    def test_returns_column_transformer(self, sample_df):
        df = sample_df.drop(columns=["HIGH_MISSING_COL"])
        preprocessor = build_preprocessor(df)
        assert isinstance(preprocessor, ColumnTransformer)

    def test_fit_transform_produces_numeric_array(self, sample_df):
        df = sample_df.drop(columns=["HIGH_MISSING_COL", "CODE_GENDER"])
        df = df[df["NAME_INCOME_TYPE"] != "Working"].reset_index(drop=True)
        preprocessor = build_preprocessor(df)
        result = preprocessor.fit_transform(df)
        assert result.shape[0] == len(df)

    def test_transform_handles_unknown_categories(self):
        train = pd.DataFrame({
            "numeric_col": [1.0, 2.0, 3.0],
            "cat_col": ["A", "B", "A"],
        })
        test = pd.DataFrame({
            "numeric_col": [4.0],
            "cat_col": ["Z"],  # unknown category
        })
        preprocessor = build_preprocessor(train)
        preprocessor.fit(train)
        result = preprocessor.transform(test)
        assert result.shape[0] == 1  # should not raise, uses unknown_value=-1

    def test_handles_nan_in_numeric(self):
        df = pd.DataFrame({
            "a": [1.0, np.nan, 3.0],
            "b": [4.0, 5.0, np.nan],
        })
        preprocessor = build_preprocessor(df)
        result = preprocessor.fit_transform(df)
        assert not np.isnan(result).any()
