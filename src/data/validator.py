"""
Data validation layer.

Checks that all tables have the expected columns and sensible row counts
before any feature engineering begins. Run once before training.
Raises ValueError with actionable messages rather than silently failing.
"""

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Minimum required columns per table (not exhaustive — just enough to catch
# truncated downloads or schema changes)
_REQUIRED_COLUMNS: dict[str, list[str]] = {
    "application": [
        "SK_ID_CURR", "TARGET", "AMT_INCOME_TOTAL", "AMT_CREDIT",
        "AMT_ANNUITY", "DAYS_BIRTH", "DAYS_EMPLOYED",
        "EXT_SOURCE_2", "EXT_SOURCE_3", "CODE_GENDER",
    ],
    "bureau": [
        "SK_ID_CURR", "SK_ID_BUREAU", "CREDIT_ACTIVE",
        "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT",
        "AMT_CREDIT_SUM_OVERDUE", "CREDIT_DAY_OVERDUE",
    ],
    "bureau_balance": ["SK_ID_BUREAU", "MONTHS_BALANCE", "STATUS"],
    "previous": [
        "SK_ID_CURR", "SK_ID_PREV", "NAME_CONTRACT_STATUS",
        "AMT_CREDIT", "AMT_ANNUITY", "DAYS_DECISION",
    ],
    "installments": [
        "SK_ID_CURR", "SK_ID_PREV", "DAYS_INSTALMENT",
        "DAYS_ENTRY_PAYMENT", "AMT_INSTALMENT", "AMT_PAYMENT",
    ],
    "pos_cash": [
        "SK_ID_CURR", "SK_ID_PREV", "MONTHS_BALANCE",
        "SK_DPD", "SK_DPD_DEF", "NAME_CONTRACT_STATUS",
    ],
    "credit_card": [
        "SK_ID_CURR", "SK_ID_PREV", "MONTHS_BALANCE",
        "AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL",
        "SK_DPD", "SK_DPD_DEF",
    ],
}

_MIN_ROWS: dict[str, int] = {
    "application":  300_000,
    "bureau":      1_000_000,
    "bureau_balance": 10_000_000,
    "previous":    1_000_000,
    "installments": 5_000_000,
    "pos_cash":    5_000_000,
    "credit_card": 1_000_000,
}


def validate_tables(tables: dict[str, pd.DataFrame]) -> None:
    """
    Validate all loaded tables.
    Raises ValueError if any table fails validation.
    Logs warnings for non-fatal issues.
    """
    errors: list[str] = []

    for name, df in tables.items():
        required_cols = _REQUIRED_COLUMNS.get(name, [])
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            errors.append(
                f"Table '{name}' is missing required columns: {missing_cols}"
            )

        min_rows = _MIN_ROWS.get(name, 0)
        if len(df) < min_rows:
            errors.append(
                f"Table '{name}' has only {len(df):,} rows (expected >= {min_rows:,}). "
                f"File may be truncated or wrong version."
            )

        null_pct = df.isnull().mean().max()
        if null_pct > 0.95:
            worst_col = df.isnull().mean().idxmax()
            logger.warning(
                f"Table '{name}' column '{worst_col}' is {null_pct:.0%} null — "
                f"likely an optional/sparse field. High-missing threshold will handle it."
            )

    if "application" in tables:
        app = tables["application"]
        if "TARGET" not in app.columns:
            errors.append("application table is missing TARGET column.")
        elif app["TARGET"].nunique() < 2:
            errors.append(
                f"TARGET column has only {app['TARGET'].nunique()} unique value(s). "
                f"Expected binary 0/1."
            )
        elif "CODE_GENDER" in app.columns:
            xna_count = (app["CODE_GENDER"] == "XNA").sum()
            if xna_count > 0:
                logger.info(
                    f"Found {xna_count} rows with CODE_GENDER=XNA — will be removed."
                )

    if errors:
        error_msg = "\n".join(f"  - {e}" for e in errors)
        raise ValueError(f"Data validation failed:\n{error_msg}")

    logger.info("All tables passed validation.")
