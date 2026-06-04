"""
Feature engineering from POS_CASH_balance.csv.

Monthly snapshots of point-of-sale and cash loan balances at Home Credit.
Reveals current-loan payment behavior and contract completion history.
"""

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_pos_cash_features(pos_cash: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate POS_CASH_balance to per-applicant (SK_ID_CURR) features.

    Returns a DataFrame with one row per SK_ID_CURR.
    """
    logger.info("Building POS cash features...")

    pos = pos_cash.copy()

    # ── Contract status flags ─────────────────────────────────────────────
    pos["IS_ACTIVE"]    = (pos["NAME_CONTRACT_STATUS"] == "Active").astype(int)
    pos["IS_COMPLETED"] = (pos["NAME_CONTRACT_STATUS"] == "Completed").astype(int)
    pos["HAS_DPD"]      = (pos["SK_DPD"] > 0).astype(int)
    pos["HAS_SEVERE_DPD"] = (pos["SK_DPD"] > 30).astype(int)

    agg = pos.groupby("SK_ID_CURR").agg(
        # ── Volume ────────────────────────────────────────────────────────
        POS_COUNT              =("SK_ID_PREV",            "count"),
        POS_UNIQUE_LOANS       =("SK_ID_PREV",            "nunique"),

        # ── Days past due ─────────────────────────────────────────────────
        POS_DPD_MAX            =("SK_DPD",               "max"),
        POS_DPD_MEAN           =("SK_DPD",               "mean"),
        POS_DPD_DEF_MAX        =("SK_DPD_DEF",           "max"),
        POS_DPD_DEF_MEAN       =("SK_DPD_DEF",           "mean"),
        POS_DPD_MONTHS         =("HAS_DPD",              "sum"),
        POS_SEVERE_DPD_MONTHS  =("HAS_SEVERE_DPD",       "sum"),
        POS_DPD_RATIO          =("HAS_DPD",              "mean"),

        # ── Contract status ───────────────────────────────────────────────
        POS_ACTIVE_COUNT       =("IS_ACTIVE",            "sum"),
        POS_COMPLETED_COUNT    =("IS_COMPLETED",         "sum"),
        POS_COMPLETED_RATIO    =("IS_COMPLETED",         "mean"),

        # ── Months balance ────────────────────────────────────────────────
        POS_MONTHS_BALANCE_MAX =("MONTHS_BALANCE",       "max"),
        POS_MONTHS_BALANCE_MEAN=("MONTHS_BALANCE",       "mean"),

        # ── Remaining installments ────────────────────────────────────────
        POS_CNT_INSTALMENT_FUTURE_MEAN =("CNT_INSTALMENT_FUTURE", "mean"),
    ).reset_index()

    logger.info(
        f"POS cash features built: {len(agg)} applicants, "
        f"{len(agg.columns) - 1} features"
    )
    return agg
