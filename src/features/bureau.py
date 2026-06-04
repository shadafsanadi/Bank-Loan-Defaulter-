"""
Feature engineering from bureau.csv and bureau_balance.csv.

Bureau data represents an applicant's credit history at OTHER financial
institutions as reported to the credit bureau — essentially a credit report.

Join path:
  bureau_balance  → group by SK_ID_BUREAU  → per-bureau-credit metrics
  bureau + above  → group by SK_ID_CURR    → per-applicant metrics
"""

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Numeric severity mapping for bureau_balance STATUS codes
# 0 = no DPD, 1-5 = 1-30 / 31-60 / 61-90 / 91-120 / 120+ DPD
# C = closed (0), X = unknown (-1)
_STATUS_MAP = {
    "C": 0, "X": -1,
    "0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5,
}


def _aggregate_bureau_balance(bureau_balance: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce bureau_balance from monthly snapshots to per-bureau-credit metrics.

    Returns a DataFrame indexed by SK_ID_BUREAU.
    """
    bb = bureau_balance.copy()
    bb["STATUS_NUMERIC"] = bb["STATUS"].map(_STATUS_MAP)
    bb["IS_DPD"] = bb["STATUS"].isin(["1", "2", "3", "4", "5"]).astype(int)
    bb["IS_SEVERE_DPD"] = bb["STATUS"].isin(["3", "4", "5"]).astype(int)

    agg = bb.groupby("SK_ID_BUREAU").agg(
        BB_MONTHS_COUNT      =("MONTHS_BALANCE", "count"),
        BB_STATUS_MAX        =("STATUS_NUMERIC", "max"),
        BB_STATUS_MEAN       =("STATUS_NUMERIC", "mean"),
        BB_DPD_MONTHS        =("IS_DPD", "sum"),
        BB_SEVERE_DPD_MONTHS =("IS_SEVERE_DPD", "sum"),
        BB_DPD_RATIO         =("IS_DPD", "mean"),
    ).reset_index()

    return agg


def build_bureau_features(
    bureau: pd.DataFrame,
    bureau_balance: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate bureau + bureau_balance into per-applicant (SK_ID_CURR) features.

    Returns a DataFrame with one row per SK_ID_CURR.
    Missing values in the result indicate the applicant has no bureau history.
    The caller should fill these with 0 before model training.
    """
    logger.info("Building bureau features...")

    bb_agg = _aggregate_bureau_balance(bureau_balance)
    bur = bureau.merge(bb_agg, on="SK_ID_BUREAU", how="left")

    # Binary loan status flags
    bur["IS_ACTIVE"]   = (bur["CREDIT_ACTIVE"] == "Active").astype(int)
    bur["IS_CLOSED"]   = (bur["CREDIT_ACTIVE"] == "Closed").astype(int)
    bur["IS_BAD_DEBT"] = (bur["CREDIT_ACTIVE"] == "Bad debt").astype(int)
    bur["IS_SOLD"]     = (bur["CREDIT_ACTIVE"] == "Sold").astype(int)

    agg = bur.groupby("SK_ID_CURR").agg(
        # ── Loan counts ──────────────────────────────────────────────────
        BUREAU_LOAN_COUNT      =("SK_ID_BUREAU", "count"),
        BUREAU_ACTIVE_COUNT    =("IS_ACTIVE",   "sum"),
        BUREAU_CLOSED_COUNT    =("IS_CLOSED",   "sum"),
        BUREAU_BAD_DEBT_COUNT  =("IS_BAD_DEBT", "sum"),
        BUREAU_SOLD_COUNT      =("IS_SOLD",     "sum"),
        BUREAU_ACTIVE_RATIO    =("IS_ACTIVE",   "mean"),

        # ── Credit amounts ────────────────────────────────────────────────
        BUREAU_CREDIT_SUM       =("AMT_CREDIT_SUM",       "sum"),
        BUREAU_CREDIT_MEAN      =("AMT_CREDIT_SUM",       "mean"),
        BUREAU_CREDIT_MAX       =("AMT_CREDIT_SUM",       "max"),
        BUREAU_DEBT_SUM         =("AMT_CREDIT_SUM_DEBT",  "sum"),
        BUREAU_DEBT_MEAN        =("AMT_CREDIT_SUM_DEBT",  "mean"),
        BUREAU_OVERDUE_SUM      =("AMT_CREDIT_SUM_OVERDUE","sum"),
        BUREAU_OVERDUE_MAX      =("AMT_CREDIT_SUM_OVERDUE","max"),
        BUREAU_LIMIT_SUM        =("AMT_CREDIT_SUM_LIMIT", "sum"),
        BUREAU_MAX_OVERDUE_EVER =("AMT_CREDIT_MAX_OVERDUE","max"),

        # ── Days past due from bureau snapshot ───────────────────────────
        BUREAU_DPD_MAX          =("CREDIT_DAY_OVERDUE",   "max"),
        BUREAU_DPD_MEAN         =("CREDIT_DAY_OVERDUE",   "mean"),

        # ── Credit timeline ───────────────────────────────────────────────
        BUREAU_DAYS_CREDIT_MEAN =("DAYS_CREDIT",          "mean"),
        BUREAU_DAYS_CREDIT_MIN  =("DAYS_CREDIT",          "min"),
        BUREAU_ENDDATE_MEAN     =("DAYS_CREDIT_ENDDATE",   "mean"),
        BUREAU_ENDDATE_MAX      =("DAYS_CREDIT_ENDDATE",   "max"),
        BUREAU_CNT_PROLONG_SUM  =("CNT_CREDIT_PROLONG",   "sum"),

        # ── Bureau balance aggregations (via merge) ───────────────────────
        BUREAU_BB_DPD_MONTHS_SUM  =("BB_DPD_MONTHS",        "sum"),
        BUREAU_BB_DPD_RATIO_MEAN  =("BB_DPD_RATIO",         "mean"),
        BUREAU_BB_STATUS_MAX      =("BB_STATUS_MAX",         "max"),
        BUREAU_BB_SEVERE_DPD_SUM  =("BB_SEVERE_DPD_MONTHS", "sum"),
        BUREAU_BB_MONTHS_MEAN     =("BB_MONTHS_COUNT",       "mean"),
    ).reset_index()

    # ── Derived ratios ────────────────────────────────────────────────────
    agg["BUREAU_DEBT_CREDIT_RATIO"] = (
        agg["BUREAU_DEBT_SUM"] / (agg["BUREAU_CREDIT_SUM"] + 1)
    )
    agg["BUREAU_OVERDUE_DEBT_RATIO"] = (
        agg["BUREAU_OVERDUE_SUM"] / (agg["BUREAU_DEBT_SUM"] + 1)
    )

    logger.info(
        f"Bureau features built: {len(agg)} applicants, "
        f"{len(agg.columns) - 1} features"
    )
    return agg
