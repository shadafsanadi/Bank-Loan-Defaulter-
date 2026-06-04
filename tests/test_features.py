"""Tests for feature engineering modules."""

import numpy as np
import pandas as pd
import pytest

from src.features.application import build_application_features
from src.features.bureau import build_bureau_features
from src.features.installments import build_installment_features


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def minimal_application():
    return pd.DataFrame({
        "SK_ID_CURR":        [1, 2, 3],
        "TARGET":            [0, 1, 0],
        "AMT_INCOME_TOTAL":  [100_000, 50_000, 200_000],
        "AMT_CREDIT":        [500_000, 300_000, 800_000],
        "AMT_ANNUITY":       [25_000, 15_000, 40_000],
        "AMT_GOODS_PRICE":   [450_000, 280_000, 750_000],
        "DAYS_BIRTH":        [-12000, -15000, -10000],
        "DAYS_EMPLOYED":     [-1000, 365243, -500],  # row 1 is unemployed sentinel
        "EXT_SOURCE_1":      [0.6, 0.3, np.nan],
        "EXT_SOURCE_2":      [0.7, 0.2, 0.9],
        "EXT_SOURCE_3":      [0.5, 0.4, 0.8],
        "CODE_GENDER":       ["F", "M", "F"],
    })


@pytest.fixture
def minimal_bureau():
    return pd.DataFrame({
        "SK_ID_CURR":             [1, 1, 2],
        "SK_ID_BUREAU":           [10, 11, 20],
        "CREDIT_ACTIVE":          ["Active", "Closed", "Bad debt"],
        "AMT_CREDIT_SUM":         [100_000, 50_000, 200_000],
        "AMT_CREDIT_SUM_DEBT":    [80_000, 0, 200_000],
        "AMT_CREDIT_SUM_OVERDUE": [0, 0, 5_000],
        "AMT_CREDIT_SUM_LIMIT":   [100_000, 50_000, 200_000],
        "AMT_CREDIT_MAX_OVERDUE": [0, 0, 10_000],
        "CREDIT_DAY_OVERDUE":     [0, 0, 30],
        "DAYS_CREDIT":            [-500, -1000, -300],
        "DAYS_CREDIT_ENDDATE":    [200, -100, 500],
        "CNT_CREDIT_PROLONG":     [0, 0, 2],
    })


@pytest.fixture
def minimal_bureau_balance():
    return pd.DataFrame({
        "SK_ID_BUREAU":   [10, 10, 11, 20, 20, 20],
        "MONTHS_BALANCE": [-1, -2, -1, -1, -2, -3],
        "STATUS":         ["0", "1", "C", "3", "2", "5"],
    })


@pytest.fixture
def minimal_installments():
    return pd.DataFrame({
        "SK_ID_CURR":              [1, 1, 1, 2, 2],
        "SK_ID_PREV":              [100, 100, 101, 200, 200],
        "NUM_INSTALMENT_VERSION":  [1, 1, 1, 1, 2],
        "NUM_INSTALMENT_NUMBER":   [1, 2, 1, 1, 1],
        "DAYS_INSTALMENT":         [-10, -40, -70, -10, -40],
        "DAYS_ENTRY_PAYMENT":      [-8, -45, -65, -15, -35],
        "AMT_INSTALMENT":          [10_000, 10_000, 10_000, 15_000, 15_000],
        "AMT_PAYMENT":             [10_000, 9_500, 10_000, 14_000, 15_000],
    })


# ── Application feature tests ─────────────────────────────────────────────────

class TestApplicationFeatures:

    def test_credit_income_ratio(self, minimal_application):
        result = build_application_features(minimal_application)
        expected = 500_000 / (100_000 + 1)
        assert abs(result["CREDIT_INCOME_RATIO"].iloc[0] - expected) < 0.01

    def test_annuity_income_ratio(self, minimal_application):
        result = build_application_features(minimal_application)
        expected = 25_000 / (100_000 + 1)
        assert abs(result["ANNUITY_INCOME_RATIO"].iloc[0] - expected) < 0.01

    def test_unemployed_sentinel_flagged(self, minimal_application):
        result = build_application_features(minimal_application)
        assert result["IS_UNEMPLOYED"].iloc[1] == 1  # row with DAYS_EMPLOYED=365243
        assert result["IS_UNEMPLOYED"].iloc[0] == 0  # normal employment

    def test_employed_years_nan_for_unemployed(self, minimal_application):
        result = build_application_features(minimal_application)
        assert pd.isna(result["EMPLOYED_YEARS"].iloc[1])

    def test_age_years_positive(self, minimal_application):
        result = build_application_features(minimal_application)
        assert (result["AGE_YEARS"] > 0).all()

    def test_ext_source_mean_handles_nan(self, minimal_application):
        result = build_application_features(minimal_application)
        # Row 2: EXT_SOURCE_1 is NaN, mean should use available values
        mean_val = result["EXT_SOURCE_MEAN"].iloc[2]
        expected = (0.9 + 0.8) / 2  # only 2 and 3 available
        assert abs(mean_val - expected) < 0.01

    def test_no_division_by_zero(self, minimal_application):
        df = minimal_application.copy()
        df["AMT_INCOME_TOTAL"] = 0  # extreme edge case
        result = build_application_features(df)
        assert not result["CREDIT_INCOME_RATIO"].isna().any()
        assert not result["ANNUITY_INCOME_RATIO"].isna().any()

    def test_output_has_more_columns_than_input(self, minimal_application):
        result = build_application_features(minimal_application)
        assert len(result.columns) > len(minimal_application.columns)


# ── Bureau feature tests ──────────────────────────────────────────────────────

class TestBureauFeatures:

    def test_returns_one_row_per_applicant(self, minimal_bureau, minimal_bureau_balance):
        result = build_bureau_features(minimal_bureau, minimal_bureau_balance)
        assert result["SK_ID_CURR"].nunique() == len(result)

    def test_bad_debt_count(self, minimal_bureau, minimal_bureau_balance):
        result = build_bureau_features(minimal_bureau, minimal_bureau_balance)
        applicant_2 = result[result["SK_ID_CURR"] == 2]
        assert applicant_2["BUREAU_BAD_DEBT_COUNT"].values[0] == 1

    def test_loan_count(self, minimal_bureau, minimal_bureau_balance):
        result = build_bureau_features(minimal_bureau, minimal_bureau_balance)
        applicant_1 = result[result["SK_ID_CURR"] == 1]
        assert applicant_1["BUREAU_LOAN_COUNT"].values[0] == 2

    def test_severe_dpd_from_bureau_balance(self, minimal_bureau, minimal_bureau_balance):
        result = build_bureau_features(minimal_bureau, minimal_bureau_balance)
        # Applicant 2 has bureau_balance STATUS 3 and 5 (both severe DPD)
        applicant_2 = result[result["SK_ID_CURR"] == 2]
        assert applicant_2["BUREAU_BB_SEVERE_DPD_SUM"].values[0] >= 2

    def test_no_negative_counts(self, minimal_bureau, minimal_bureau_balance):
        result = build_bureau_features(minimal_bureau, minimal_bureau_balance)
        count_cols = [c for c in result.columns if "COUNT" in c or "SUM" in c]
        for col in count_cols:
            assert (result[col] >= 0).all(), f"{col} has negative values"


# ── Installment feature tests ─────────────────────────────────────────────────

class TestInstallmentFeatures:

    def test_returns_one_row_per_applicant(self, minimal_installments):
        result = build_installment_features(minimal_installments)
        assert result["SK_ID_CURR"].nunique() == len(result)

    def test_late_payment_detection(self, minimal_installments):
        result = build_installment_features(minimal_installments)
        # Applicant 2: row 0 paid 5 days late (DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT = -15 - (-10) = -5 → late)
        # Actually: DAYS_ENTRY_PAYMENT=-15, DAYS_INSTALMENT=-10 → DPD = -15 - (-10) = -5 → early
        # Row 1: DAYS_ENTRY_PAYMENT=-35, DAYS_INSTALMENT=-40 → DPD = -35 - (-40) = 5 → late
        applicant_2 = result[result["SK_ID_CURR"] == 2]
        assert applicant_2["INSTAL_LATE_COUNT"].values[0] >= 1

    def test_underpayment_detection(self, minimal_installments):
        result = build_installment_features(minimal_installments)
        # Applicant 1 row 1: paid 9500 vs 10000 owed → underpayment
        applicant_1 = result[result["SK_ID_CURR"] == 1]
        assert applicant_1["INSTAL_UNDERPAYMENT_COUNT"].values[0] >= 1

    def test_dpd_max_capped_at_365(self, minimal_installments):
        ins = minimal_installments.copy()
        ins.loc[0, "DAYS_ENTRY_PAYMENT"] = ins.loc[0, "DAYS_INSTALMENT"] + 1000
        result = build_installment_features(ins)
        assert result["INSTAL_DPD_MAX"].max() <= 365
