"""
Feature engineering from previous_application.csv.

Captures the applicant's history of past loan applications at Home Credit
specifically. This reveals approval/refusal patterns, loan type preferences,
and historical credit amounts requested.
"""

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_previous_features(previous: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate previous_application to per-applicant (SK_ID_CURR) features.

    Returns a DataFrame with one row per SK_ID_CURR.
    """
    logger.info("Building previous application features...")

    prev = previous.copy()

    # ── Contract status flags ─────────────────────────────────────────────
    prev["IS_APPROVED"]  = (prev["NAME_CONTRACT_STATUS"] == "Approved").astype(int)
    prev["IS_REFUSED"]   = (prev["NAME_CONTRACT_STATUS"] == "Refused").astype(int)
    prev["IS_CANCELLED"] = (prev["NAME_CONTRACT_STATUS"] == "Canceled").astype(int)
    prev["IS_UNUSED"]    = (prev["NAME_CONTRACT_STATUS"] == "Unused offer").astype(int)

    # ── Loan type flags ───────────────────────────────────────────────────
    prev["IS_CONSUMER_LOAN"] = (prev["NAME_CONTRACT_TYPE"] == "Consumer loans").astype(int)
    prev["IS_CASH_LOAN"]     = (prev["NAME_CONTRACT_TYPE"] == "Cash loans").astype(int)
    prev["IS_REVOLVING"]     = (prev["NAME_CONTRACT_TYPE"] == "Revolving loans").astype(int)

    # ── Late / early application indicators ───────────────────────────────
    # DAYS_DECISION: negative = in the past, 0 = same day as current application
    prev["WAS_RECENT_APP"] = (prev["DAYS_DECISION"] >= -365).astype(int)  # within last year

    agg = prev.groupby("SK_ID_CURR").agg(
        # ── Application counts ────────────────────────────────────────────
        PREV_APP_COUNT         =("SK_ID_PREV",     "count"),
        PREV_APPROVED_COUNT    =("IS_APPROVED",    "sum"),
        PREV_REFUSED_COUNT     =("IS_REFUSED",     "sum"),
        PREV_CANCELLED_COUNT   =("IS_CANCELLED",   "sum"),
        PREV_UNUSED_COUNT      =("IS_UNUSED",      "sum"),
        PREV_APPROVED_RATIO    =("IS_APPROVED",    "mean"),
        PREV_REFUSED_RATIO     =("IS_REFUSED",     "mean"),
        PREV_RECENT_APP_COUNT  =("WAS_RECENT_APP", "sum"),

        # ── Loan types ────────────────────────────────────────────────────
        PREV_CONSUMER_LOAN_COUNT =("IS_CONSUMER_LOAN", "sum"),
        PREV_CASH_LOAN_COUNT     =("IS_CASH_LOAN",     "sum"),
        PREV_REVOLVING_COUNT     =("IS_REVOLVING",     "sum"),

        # ── Credit amounts ────────────────────────────────────────────────
        PREV_AMT_CREDIT_MEAN   =("AMT_CREDIT",          "mean"),
        PREV_AMT_CREDIT_MAX    =("AMT_CREDIT",          "max"),
        PREV_AMT_CREDIT_SUM    =("AMT_CREDIT",          "sum"),
        PREV_AMT_ANNUITY_MEAN  =("AMT_ANNUITY",         "mean"),
        PREV_AMT_APPLICATION_MEAN =("AMT_APPLICATION",  "mean"),

        # ── Down payment ──────────────────────────────────────────────────
        # Higher down payment rate = more skin in the game = lower risk
        PREV_RATE_DOWN_PAYMENT_MEAN =("RATE_DOWN_PAYMENT", "mean"),
        PREV_RATE_DOWN_PAYMENT_MAX  =("RATE_DOWN_PAYMENT", "max"),

        # ── Installment counts ────────────────────────────────────────────
        PREV_CNT_PAYMENT_MEAN  =("CNT_PAYMENT", "mean"),
        PREV_CNT_PAYMENT_MAX   =("CNT_PAYMENT", "max"),

        # ── Decision timeline ─────────────────────────────────────────────
        PREV_DAYS_DECISION_MEAN =("DAYS_DECISION", "mean"),
        PREV_DAYS_DECISION_MAX  =("DAYS_DECISION", "max"),  # most recent app
    ).reset_index()

    # ── Derived ratios ────────────────────────────────────────────────────
    # Credit granted vs. applied — consistently receiving less than requested
    # may indicate the borrower repeatedly overestimates creditworthiness
    agg["PREV_CREDIT_RATIO"] = (
        agg["PREV_AMT_CREDIT_MEAN"] / (agg["PREV_AMT_APPLICATION_MEAN"] + 1)
    )

    logger.info(
        f"Previous features built: {len(agg)} applicants, "
        f"{len(agg.columns) - 1} features"
    )
    return agg
