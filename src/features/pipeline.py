"""
Master feature pipeline.

Orchestrates loading all tables, running each feature module, and
joining the results into a single master DataFrame (one row per applicant).

This is the entry point for both training and batch inference.
Output: data/processed/master_train.csv
"""

from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger
from src.utils.paths import get_path
from src.data.loader import DataLoader
from src.data.validator import validate_tables
from src.features.application import build_application_features
from src.features.bureau import build_bureau_features
from src.features.previous import build_previous_features
from src.features.installments import build_installment_features
from src.features.pos_cash import build_pos_cash_features
from src.features.credit_card import build_credit_card_features

logger = get_logger(__name__)


def build_master_features(
    tables: dict[str, pd.DataFrame],
    save: bool = False,
) -> pd.DataFrame:
    """
    Build the master feature table by joining all feature modules.

    Args:
        tables: dict returned by DataLoader.load_all()
        save:   If True, save the result to data/processed/master_train.csv

    Returns:
        Master DataFrame with one row per SK_ID_CURR and all engineered features.
        Rows are in the same order as the application table.
    """
    logger.info("Building master feature table...")

    # ── Step 1: Application features (main table) ─────────────────────────
    master = build_application_features(tables["application"])

    # ── Step 2: Bureau features ───────────────────────────────────────────
    bureau_feats = build_bureau_features(
        tables["bureau"], tables["bureau_balance"]
    )
    master = master.merge(bureau_feats, on="SK_ID_CURR", how="left")

    # ── Step 3: Previous application features ─────────────────────────────
    prev_feats = build_previous_features(tables["previous"])
    master = master.merge(prev_feats, on="SK_ID_CURR", how="left")

    # ── Step 4: Installment payment features ──────────────────────────────
    instal_feats = build_installment_features(tables["installments"])
    master = master.merge(instal_feats, on="SK_ID_CURR", how="left")

    # ── Step 5: POS cash features ─────────────────────────────────────────
    pos_feats = build_pos_cash_features(tables["pos_cash"])
    master = master.merge(pos_feats, on="SK_ID_CURR", how="left")

    # ── Step 6: Credit card features ──────────────────────────────────────
    cc_feats = build_credit_card_features(tables["credit_card"])
    master = master.merge(cc_feats, on="SK_ID_CURR", how="left")

    # ── Step 7: Fill NaN from left joins ──────────────────────────────────
    # Applicants with no history in supplementary tables get NaN values.
    # For behavioral features, NaN means "no history" which is different from
    # "average history", so we fill with 0 rather than median for these columns.
    bureau_cols   = [c for c in master.columns if c.startswith("BUREAU_")]
    prev_cols     = [c for c in master.columns if c.startswith("PREV_")]
    instal_cols   = [c for c in master.columns if c.startswith("INSTAL_")]
    pos_cols      = [c for c in master.columns if c.startswith("POS_")]
    cc_cols       = [c for c in master.columns if c.startswith("CC_")]

    zero_fill_cols = bureau_cols + prev_cols + instal_cols + pos_cols + cc_cols
    master[zero_fill_cols] = master[zero_fill_cols].fillna(0)

    logger.info(
        f"Master feature table built: {len(master):,} rows, "
        f"{len(master.columns)} columns"
    )

    if save:
        out_path = get_path("data", "processed", "master_train.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        master.to_csv(out_path, index=False)
        logger.info(f"Master table saved to {out_path}")

    return master


def load_or_build_master(
    raw_dir: str | Path | None = None,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """
    Return the master feature table, loading from cache if available.

    Args:
        raw_dir:       Path to raw CSV files. Defaults to data/raw/.
        force_rebuild: If True, always rebuild from raw files even if cache exists.
    """
    cache_path = get_path("data", "processed", "master_train.csv")

    if cache_path.exists() and not force_rebuild:
        logger.info(f"Loading master table from cache: {cache_path}")
        return pd.read_csv(cache_path)

    logger.info("No cache found (or force_rebuild=True). Building from raw files...")
    raw_dir = raw_dir or get_path("data", "raw")
    loader = DataLoader(raw_dir)
    tables = loader.load_all()
    validate_tables(tables)
    return build_master_features(tables, save=True)
