"""
Feature engineering from credit_card_balance.csv.

Monthly credit card account snapshots revealing credit utilization,
payment behavior, and cash advance patterns.

Credit utilization (balance / limit) is one of the strongest predictors of
default risk in consumer credit — high utilization signals financial stress.
"""

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_credit_card_features(credit_card: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate credit_card_balance to per-applicant (SK_ID_CURR) features.

    Returns a DataFrame with one row per SK_ID_CURR.
    """
    logger.info("Building credit card features...")

    cc = credit_card.copy()

    # ── Utilization ───────────────────────────────────────────────────────
    # Core signal: what fraction of available credit is being used?
    cc["UTILIZATION"] = cc["AMT_BALANCE"] / (cc["AMT_CREDIT_LIMIT_ACTUAL"] + 1)
    cc["UTILIZATION"] = cc["UTILIZATION"].clip(0, 1)  # cap at 100%; >100% = data error

    # ── Payment behavior ──────────────────────────────────────────────────
    # Did the client pay more than the minimum required?
    cc["PAYS_ABOVE_MINIMUM"] = (
        cc["AMT_PAYMENT_TOTAL_CURRENT"] > cc["AMT_INST_MIN_REGULARITY"]
    ).astype(int)

    # Payment rate vs. balance — 1.0 = full payoff, <0.1 = minimum-only payer
    cc["PAYMENT_RATE"] = cc["AMT_PAYMENT_TOTAL_CURRENT"] / (cc["AMT_BALANCE"] + 1)
    cc["PAYMENT_RATE"] = cc["PAYMENT_RATE"].clip(0, 2)  # >2 = overpayment, artifact

    # ── DPD flags ─────────────────────────────────────────────────────────
    cc["HAS_DPD"] = (cc["SK_DPD"] > 0).astype(int)
    cc["HAS_SEVERE_DPD"] = (cc["SK_DPD"] > 30).astype(int)

    agg = cc.groupby("SK_ID_CURR").agg(
        # ── Volume ────────────────────────────────────────────────────────
        CC_COUNT               =("SK_ID_PREV",                     "count"),
        CC_UNIQUE_CARDS        =("SK_ID_PREV",                     "nunique"),

        # ── Utilization ───────────────────────────────────────────────────
        CC_UTILIZATION_MEAN    =("UTILIZATION",                    "mean"),
        CC_UTILIZATION_MAX     =("UTILIZATION",                    "max"),
        CC_UTILIZATION_STD     =("UTILIZATION",                    "std"),

        # ── Balances ──────────────────────────────────────────────────────
        CC_AMT_BALANCE_MEAN    =("AMT_BALANCE",                    "mean"),
        CC_AMT_BALANCE_MAX     =("AMT_BALANCE",                    "max"),
        CC_CREDIT_LIMIT_MEAN   =("AMT_CREDIT_LIMIT_ACTUAL",        "mean"),

        # ── Payment behavior ──────────────────────────────────────────────
        CC_PAYMENT_RATE_MEAN   =("PAYMENT_RATE",                   "mean"),
        CC_PAYMENT_RATE_MIN    =("PAYMENT_RATE",                   "min"),
        CC_PAYS_ABOVE_MIN_RATIO=("PAYS_ABOVE_MINIMUM",             "mean"),
        CC_AMT_PAYMENT_MEAN    =("AMT_PAYMENT_TOTAL_CURRENT",      "mean"),

        # ── Cash advance behavior ─────────────────────────────────────────
        # ATM cash draws are expensive and signal financial distress
        CC_DRAWINGS_ATM_MEAN   =("AMT_DRAWINGS_ATM_CURRENT",       "mean"),
        CC_DRAWINGS_TOTAL_MEAN =("AMT_DRAWINGS_CURRENT",           "mean"),
        CC_CNT_DRAWINGS_ATM    =("CNT_DRAWINGS_ATM_CURRENT",       "mean"),

        # ── Days past due ─────────────────────────────────────────────────
        CC_DPD_MAX             =("SK_DPD",                         "max"),
        CC_DPD_MEAN            =("SK_DPD",                         "mean"),
        CC_DPD_DEF_MAX         =("SK_DPD_DEF",                     "max"),
        CC_DPD_MONTHS          =("HAS_DPD",                        "sum"),
        CC_SEVERE_DPD_MONTHS   =("HAS_SEVERE_DPD",                 "sum"),
        CC_DPD_RATIO           =("HAS_DPD",                        "mean"),
    ).reset_index()

    # ATM draw ratio: cash draws as fraction of total draws — high ratio = distress
    agg["CC_ATM_DRAW_RATIO"] = (
        agg["CC_DRAWINGS_ATM_MEAN"] / (agg["CC_DRAWINGS_TOTAL_MEAN"] + 1)
    )

    logger.info(
        f"Credit card features built: {len(agg)} applicants, "
        f"{len(agg.columns) - 1} features"
    )
    return agg
