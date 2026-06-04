"""
Feature engineering from application_train.csv.

These are features derivable at loan application time from the applicant's
own declared information. They represent the "first impression" risk signals
available before any bureau lookup.
"""

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_application_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features to the application DataFrame.

    All formulas use +1 in denominators to prevent division by zero.
    The additive offset is negligible at the scale of these financial values.
    """
    df = df.copy()
    logger.info("Building application features...")

    # ── Credit burden ratios ────────────────────────────────────────────────
    # Higher ratio = more debt relative to income = higher financial strain
    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / (df["AMT_INCOME_TOTAL"] + 1)

    # Monthly installment as a fraction of income — the core affordability signal
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] + 1)

    # Loan duration in months — longer exposure = more opportunity to default
    df["CREDIT_TERM_MONTHS"] = df["AMT_CREDIT"] / (df["AMT_ANNUITY"] + 1)

    # Goods value vs. credit — gap indicates additional financed costs (insurance, fees)
    df["GOODS_CREDIT_RATIO"] = df["AMT_GOODS_PRICE"] / (df["AMT_CREDIT"] + 1)

    # ── Age and employment ──────────────────────────────────────────────────
    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365

    # DAYS_EMPLOYED = 365243 is the dataset's sentinel for "not employed"
    # Must be handled before computing EMPLOYED_YEARS or it creates ~1000-year values
    df["IS_UNEMPLOYED"] = (df["DAYS_EMPLOYED"] == 365243).astype(int)
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)
    df["EMPLOYED_YEARS"] = -df["DAYS_EMPLOYED"] / 365

    # Fraction of career spent at current employer — stability signal
    df["EMPLOYMENT_AGE_RATIO"] = df["EMPLOYED_YEARS"] / (df["AGE_YEARS"] + 1)

    # Years since registration — long registration = more stable address history
    df["REGISTRATION_YEARS"] = -df.get("DAYS_REGISTRATION", pd.Series(0, index=df.index)) / 365

    # Years since last ID document publication — very recent renewal may indicate
    # identity instability
    df["ID_PUBLISH_YEARS"] = -df.get("DAYS_ID_PUBLISH", pd.Series(0, index=df.index)) / 365

    # ── External credit scores ──────────────────────────────────────────────
    # These are the single most predictive features in the dataset
    ext_cols = [c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"] if c in df.columns]
    if ext_cols:
        df["EXT_SOURCE_MEAN"] = df[ext_cols].mean(axis=1)
        df["EXT_SOURCE_MIN"] = df[ext_cols].min(axis=1)
        df["EXT_SOURCE_STD"] = df[ext_cols].std(axis=1).fillna(0)

    if "EXT_SOURCE_2" in df.columns and "EXT_SOURCE_3" in df.columns:
        # Product captures cases where BOTH scores must be good (multiplicative risk)
        df["EXT_SOURCE_PRODUCT"] = df["EXT_SOURCE_2"] * df["EXT_SOURCE_3"]

    # Interaction: high debt burden + low external score = compounded risk
    if "EXT_SOURCE_2" in df.columns:
        df["CREDIT_INCOME_x_EXT2"] = df["CREDIT_INCOME_RATIO"] * df["EXT_SOURCE_2"]
        df["ANNUITY_INCOME_x_EXT2"] = df["ANNUITY_INCOME_RATIO"] * df["EXT_SOURCE_2"]

    # ── Document and contact flags ──────────────────────────────────────────
    flag_doc_cols = [c for c in df.columns if c.startswith("FLAG_DOCUMENT_")]
    if flag_doc_cols:
        df["DOCS_PROVIDED_COUNT"] = df[flag_doc_cols].sum(axis=1)

    contact_cols = [
        c for c in ["FLAG_MOBIL", "FLAG_EMP_PHONE", "FLAG_WORK_PHONE",
                    "FLAG_CONT_MOBILE", "FLAG_PHONE", "FLAG_EMAIL"]
        if c in df.columns
    ]
    if contact_cols:
        df["CONTACT_CHANNELS_COUNT"] = df[contact_cols].sum(axis=1)

    # ── Car and realty ownership ────────────────────────────────────────────
    if "FLAG_OWN_CAR" in df.columns:
        df["FLAG_OWN_CAR"] = (df["FLAG_OWN_CAR"] == "Y").astype(int)
    if "FLAG_OWN_REALTY" in df.columns:
        df["FLAG_OWN_REALTY"] = (df["FLAG_OWN_REALTY"] == "Y").astype(int)

    n_new = len([c for c in df.columns if c not in [
        "SK_ID_CURR", "TARGET", "AMT_INCOME_TOTAL", "AMT_CREDIT",
        "AMT_ANNUITY", "AMT_GOODS_PRICE", "DAYS_BIRTH", "DAYS_EMPLOYED",
    ]])
    logger.info(f"Application features built. Total columns: {len(df.columns)}")
    return df
