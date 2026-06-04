"""
Data loading layer.

Single responsibility: read raw CSV files from data/raw/ and return DataFrames.
No transformation logic lives here — all transformations are in src/features/.
"""

from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Expected filenames — centralised so a rename only needs updating here
_FILES = {
    "application":       "application_train.csv",
    "bureau":            "bureau.csv",
    "bureau_balance":    "bureau_balance.csv",
    "previous":          "previous_application.csv",
    "installments":      "installments_payments.csv",
    "pos_cash":          "POS_CASH_balance.csv",
    "credit_card":       "credit_card_balance.csv",
}


class DataLoader:
    """Loads all Home Credit dataset tables from a local directory."""

    def __init__(self, raw_dir: str | Path):
        self.raw_dir = Path(raw_dir)

    def _read(self, key: str) -> pd.DataFrame:
        filename = _FILES[key]
        path = self.raw_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Expected data file not found: {path}\n"
                f"Download the Home Credit dataset from Kaggle and place CSVs in {self.raw_dir}"
            )
        size_mb = path.stat().st_size / 1_000_000
        logger.info(f"Loading {filename} ({size_mb:.1f} MB)")
        return pd.read_csv(path)

    def load_application(self) -> pd.DataFrame:
        return self._read("application")

    def load_bureau(self) -> pd.DataFrame:
        return self._read("bureau")

    def load_bureau_balance(self) -> pd.DataFrame:
        return self._read("bureau_balance")

    def load_previous(self) -> pd.DataFrame:
        return self._read("previous")

    def load_installments(self) -> pd.DataFrame:
        return self._read("installments")

    def load_pos_cash(self) -> pd.DataFrame:
        return self._read("pos_cash")

    def load_credit_card(self) -> pd.DataFrame:
        return self._read("credit_card")

    def load_all(self) -> dict[str, pd.DataFrame]:
        """Load all 7 tables and return as a named dict."""
        tables = {}
        for key in _FILES:
            tables[key] = self._read(key)
        logger.info(
            "All tables loaded. Row counts: "
            + ", ".join(f"{k}={len(v):,}" for k, v in tables.items())
        )
        return tables
