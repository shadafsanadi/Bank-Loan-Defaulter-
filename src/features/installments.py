"""
Feature engineering from installments_payments.csv.

This table contains the actual repayment records for all previous Home Credit
loans — the most behaviorally rich signal in the dataset.

Key insight: the difference between DAYS_ENTRY_PAYMENT and DAYS_INSTALMENT
captures payment lateness (positive = late), and AMT_PAYMENT vs. AMT_INSTALMENT
captures underpayment behavior.
"""

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_installment_features(installments: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate installments_payments to per-applicant (SK_ID_CURR) features.

    Returns a DataFrame with one row per SK_ID_CURR.
    """
    logger.info("Building installment features...")

    ins = installments.copy()

    # ── Payment timing (days past due per payment) ─────────────────────────
    # Positive value = late payment, negative = early
    ins["PAYMENT_DPD"] = ins["DAYS_ENTRY_PAYMENT"] - ins["DAYS_INSTALMENT"]
    ins["IS_LATE"] = (ins["PAYMENT_DPD"] > 0).astype(int)

    # ── Payment amount deficit ─────────────────────────────────────────────
    # Positive deficit = underpayment
    ins["PAYMENT_DEFICIT"] = ins["AMT_INSTALMENT"] - ins["AMT_PAYMENT"]
    ins["IS_UNDERPAYMENT"] = (ins["PAYMENT_DEFICIT"] > 0).astype(int)

    # Payment as fraction of installment due
    ins["PAYMENT_PERC"] = ins["AMT_PAYMENT"] / (ins["AMT_INSTALMENT"] + 1)

    agg = ins.groupby("SK_ID_CURR").agg(
        # ── Volume ───────────────────────────────────────────────────────
        INSTAL_COUNT                   =("SK_ID_PREV", "count"),

        # ── Days past due ─────────────────────────────────────────────────
        INSTAL_DPD_MAX                 =("PAYMENT_DPD", "max"),
        INSTAL_DPD_MEAN                =("PAYMENT_DPD", "mean"),
        INSTAL_DPD_STD                 =("PAYMENT_DPD", "std"),
        INSTAL_LATE_COUNT              =("IS_LATE",      "sum"),
        INSTAL_LATE_RATIO              =("IS_LATE",      "mean"),

        # ── Payment amount ────────────────────────────────────────────────
        INSTAL_AMT_PAYMENT_MEAN        =("AMT_PAYMENT",   "mean"),
        INSTAL_AMT_PAYMENT_SUM         =("AMT_PAYMENT",   "sum"),
        INSTAL_DEFICIT_MEAN            =("PAYMENT_DEFICIT","mean"),
        INSTAL_DEFICIT_SUM             =("PAYMENT_DEFICIT","sum"),
        INSTAL_DEFICIT_MAX             =("PAYMENT_DEFICIT","max"),
        INSTAL_UNDERPAYMENT_COUNT      =("IS_UNDERPAYMENT","sum"),
        INSTAL_UNDERPAYMENT_RATIO      =("IS_UNDERPAYMENT","mean"),

        # ── Payment fraction ──────────────────────────────────────────────
        INSTAL_PAYMENT_PERC_MEAN       =("PAYMENT_PERC",  "mean"),
        INSTAL_PAYMENT_PERC_MIN        =("PAYMENT_PERC",  "min"),

        # ── Installment restructuring ─────────────────────────────────────
        # Loans with many version changes were renegotiated — potential distress signal
        INSTAL_VERSION_MAX             =("NUM_INSTALMENT_VERSION", "max"),
        INSTAL_NUM_UNIQUE_LOANS        =("SK_ID_PREV", "nunique"),
    ).reset_index()

    # Cap DPD max to remove measurement errors (DPD > 365 is implausible)
    agg["INSTAL_DPD_MAX"] = agg["INSTAL_DPD_MAX"].clip(upper=365)

    # Standardise deficit as ratio for scale-invariant comparison across loan sizes
    agg["INSTAL_DEFICIT_MEAN"] = agg["INSTAL_DEFICIT_MEAN"].fillna(0)

    logger.info(
        f"Installment features built: {len(agg)} applicants, "
        f"{len(agg.columns) - 1} features"
    )
    return agg
