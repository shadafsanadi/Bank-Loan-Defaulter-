"""Integration-style tests for the feature pipeline and data loader."""

import numpy as np
import pandas as pd
import pytest

from src.features.pipeline import build_master_features
from src.features.previous import build_previous_features
from src.features.pos_cash import build_pos_cash_features
from src.features.credit_card import build_credit_card_features


@pytest.fixture
def minimal_tables():
    """Minimal set of tables sufficient to test build_master_features."""
    application = pd.DataFrame({
        "SK_ID_CURR":        [1, 2, 3],
        "TARGET":            [0, 1, 0],
        "AMT_INCOME_TOTAL":  [100_000, 50_000, 200_000],
        "AMT_CREDIT":        [400_000, 300_000, 800_000],
        "AMT_ANNUITY":       [20_000, 15_000, 40_000],
        "AMT_GOODS_PRICE":   [380_000, 280_000, 760_000],
        "DAYS_BIRTH":        [-12_000, -15_000, -10_000],
        "DAYS_EMPLOYED":     [-1_000, 365_243, -500],
        "EXT_SOURCE_1":      [0.6, np.nan, 0.8],
        "EXT_SOURCE_2":      [0.7, 0.2, 0.9],
        "EXT_SOURCE_3":      [0.5, 0.4, 0.8],
        "CODE_GENDER":       ["F", "M", "F"],
    })
    bureau = pd.DataFrame({
        "SK_ID_CURR":             [1, 2],
        "SK_ID_BUREAU":           [10, 20],
        "CREDIT_ACTIVE":          ["Active", "Closed"],
        "AMT_CREDIT_SUM":         [100_000, 50_000],
        "AMT_CREDIT_SUM_DEBT":    [80_000, 0],
        "AMT_CREDIT_SUM_OVERDUE": [0, 0],
        "AMT_CREDIT_SUM_LIMIT":   [100_000, 50_000],
        "AMT_CREDIT_MAX_OVERDUE": [0, 0],
        "CREDIT_DAY_OVERDUE":     [0, 0],
        "DAYS_CREDIT":            [-500, -1_000],
        "DAYS_CREDIT_ENDDATE":    [200, -100],
        "CNT_CREDIT_PROLONG":     [0, 0],
    })
    bureau_balance = pd.DataFrame({
        "SK_ID_BUREAU":   [10, 10, 20],
        "MONTHS_BALANCE": [-1, -2, -1],
        "STATUS":         ["0", "1", "C"],
    })
    previous = pd.DataFrame({
        "SK_ID_CURR":           [1, 1, 2],
        "SK_ID_PREV":           [100, 101, 200],
        "NAME_CONTRACT_STATUS": ["Approved", "Refused", "Approved"],
        "NAME_CONTRACT_TYPE":   ["Cash loans", "Consumer loans", "Cash loans"],
        "AMT_CREDIT":           [300_000, 250_000, 200_000],
        "AMT_ANNUITY":          [15_000, 12_000, 10_000],
        "AMT_APPLICATION":      [300_000, 280_000, 200_000],
        "RATE_DOWN_PAYMENT":    [0.1, np.nan, 0.15],
        "CNT_PAYMENT":          [24, 18, 12],
        "DAYS_DECISION":        [-100, -200, -50],
    })
    installments = pd.DataFrame({
        "SK_ID_CURR":             [1, 1, 2],
        "SK_ID_PREV":             [100, 100, 200],
        "NUM_INSTALMENT_VERSION": [1, 1, 1],
        "NUM_INSTALMENT_NUMBER":  [1, 2, 1],
        "DAYS_INSTALMENT":        [-10, -40, -10],
        "DAYS_ENTRY_PAYMENT":     [-8, -45, -15],
        "AMT_INSTALMENT":         [10_000, 10_000, 15_000],
        "AMT_PAYMENT":            [10_000, 9_800, 14_000],
    })
    pos_cash = pd.DataFrame({
        "SK_ID_CURR":           [1, 1, 2],
        "SK_ID_PREV":           [100, 100, 200],
        "MONTHS_BALANCE":       [-1, -2, -1],
        "NAME_CONTRACT_STATUS": ["Active", "Active", "Completed"],
        "SK_DPD":               [0, 0, 5],
        "SK_DPD_DEF":           [0, 0, 0],
        "CNT_INSTALMENT_FUTURE":[10, 11, 0],
    })
    credit_card = pd.DataFrame({
        "SK_ID_CURR":                    [1, 1],
        "SK_ID_PREV":                    [100, 100],
        "MONTHS_BALANCE":                [-1, -2],
        "AMT_BALANCE":                   [50_000, 40_000],
        "AMT_CREDIT_LIMIT_ACTUAL":       [100_000, 100_000],
        "AMT_DRAWINGS_ATM_CURRENT":      [5_000, 3_000],
        "AMT_DRAWINGS_CURRENT":          [10_000, 8_000],
        "AMT_INST_MIN_REGULARITY":       [2_000, 2_000],
        "AMT_PAYMENT_TOTAL_CURRENT":     [10_000, 8_000],
        "CNT_DRAWINGS_ATM_CURRENT":      [2, 1],
        "NAME_CONTRACT_STATUS":          ["Active", "Active"],
        "SK_DPD":                        [0, 0],
        "SK_DPD_DEF":                    [0, 0],
    })
    return {
        "application":   application,
        "bureau":        bureau,
        "bureau_balance":bureau_balance,
        "previous":      previous,
        "installments":  installments,
        "pos_cash":      pos_cash,
        "credit_card":   credit_card,
    }


class TestMasterPipeline:

    def test_output_has_same_row_count_as_application(self, minimal_tables):
        result = build_master_features(minimal_tables)
        assert len(result) == len(minimal_tables["application"])

    def test_sk_id_curr_preserved(self, minimal_tables):
        result = build_master_features(minimal_tables)
        assert set(result["SK_ID_CURR"]) == {1, 2, 3}

    def test_target_preserved(self, minimal_tables):
        result = build_master_features(minimal_tables)
        assert "TARGET" in result.columns
        assert set(result["TARGET"]) == {0, 1}

    def test_applicant_with_no_bureau_gets_zero_fill(self, minimal_tables):
        result = build_master_features(minimal_tables)
        # Applicant 3 has no bureau records
        applicant_3 = result[result["SK_ID_CURR"] == 3]
        assert applicant_3["BUREAU_LOAN_COUNT"].values[0] == 0

    def test_applicant_with_no_credit_card_gets_zero_fill(self, minimal_tables):
        result = build_master_features(minimal_tables)
        applicant_2 = result[result["SK_ID_CURR"] == 2]
        assert applicant_2["CC_COUNT"].values[0] == 0

    def test_no_duplicate_rows(self, minimal_tables):
        result = build_master_features(minimal_tables)
        assert result["SK_ID_CURR"].nunique() == len(result)

    def test_application_features_present(self, minimal_tables):
        result = build_master_features(minimal_tables)
        for col in ["CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "AGE_YEARS",
                    "IS_UNEMPLOYED", "EXT_SOURCE_MEAN"]:
            assert col in result.columns, f"Missing: {col}"

    def test_bureau_features_present(self, minimal_tables):
        result = build_master_features(minimal_tables)
        for col in ["BUREAU_LOAN_COUNT", "BUREAU_ACTIVE_COUNT", "BUREAU_OVERDUE_SUM"]:
            assert col in result.columns, f"Missing: {col}"

    def test_installment_features_present(self, minimal_tables):
        result = build_master_features(minimal_tables)
        for col in ["INSTAL_DPD_MAX", "INSTAL_LATE_RATIO", "INSTAL_PAYMENT_PERC_MEAN"]:
            assert col in result.columns, f"Missing: {col}"


class TestPreviousFeatures:

    def test_approved_ratio_correct(self):
        prev = pd.DataFrame({
            "SK_ID_CURR":           [1, 1, 1],
            "SK_ID_PREV":           [10, 11, 12],
            "NAME_CONTRACT_STATUS": ["Approved", "Refused", "Approved"],
            "NAME_CONTRACT_TYPE":   ["Cash loans", "Cash loans", "Cash loans"],
            "AMT_CREDIT":           [100_000, 80_000, 120_000],
            "AMT_ANNUITY":          [5_000, 4_000, 6_000],
            "AMT_APPLICATION":      [100_000, 90_000, 120_000],
            "RATE_DOWN_PAYMENT":    [0.1, 0.1, 0.15],
            "CNT_PAYMENT":          [24, 18, 12],
            "DAYS_DECISION":        [-100, -200, -50],
        })
        result = build_previous_features(prev)
        assert abs(result["PREV_APPROVED_RATIO"].values[0] - 2/3) < 0.01

    def test_cash_loan_count(self):
        prev = pd.DataFrame({
            "SK_ID_CURR":           [1, 1],
            "SK_ID_PREV":           [10, 11],
            "NAME_CONTRACT_STATUS": ["Approved", "Approved"],
            "NAME_CONTRACT_TYPE":   ["Cash loans", "Consumer loans"],
            "AMT_CREDIT":           [100_000, 80_000],
            "AMT_ANNUITY":          [5_000, 4_000],
            "AMT_APPLICATION":      [100_000, 80_000],
            "RATE_DOWN_PAYMENT":    [0.1, 0.1],
            "CNT_PAYMENT":          [24, 18],
            "DAYS_DECISION":        [-100, -200],
        })
        result = build_previous_features(prev)
        assert result["PREV_CASH_LOAN_COUNT"].values[0] == 1


class TestCreditCardFeatures:

    def test_utilization_capped_at_1(self):
        cc = pd.DataFrame({
            "SK_ID_CURR":               [1],
            "SK_ID_PREV":               [10],
            "MONTHS_BALANCE":           [-1],
            "AMT_BALANCE":              [200_000],   # Over limit
            "AMT_CREDIT_LIMIT_ACTUAL":  [100_000],
            "AMT_DRAWINGS_ATM_CURRENT": [0],
            "AMT_DRAWINGS_CURRENT":     [0],
            "AMT_INST_MIN_REGULARITY":  [2_000],
            "AMT_PAYMENT_TOTAL_CURRENT":[2_000],
            "CNT_DRAWINGS_ATM_CURRENT": [0],
            "NAME_CONTRACT_STATUS":     ["Active"],
            "SK_DPD":                   [0],
            "SK_DPD_DEF":               [0],
        })
        result = build_credit_card_features(cc)
        assert result["CC_UTILIZATION_MEAN"].values[0] <= 1.0
