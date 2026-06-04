"""
Post-training SHAP analysis script.

Run this after `python src/models/train.py` to generate:
  - outputs/plots/shap_global_importance.png
  - outputs/plots/shap_summary.png
  - outputs/feature_importance.csv
  - models/v2/shap_explainer.pkl

Usage:
    python src/models/run_explain.py
    # or: make explain
"""

import sys
from pathlib import Path

import joblib
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.paths import get_path, get_config_path
from src.utils.logger import get_logger
from src.features.pipeline import load_or_build_master
from src.preprocessing import remove_invalid_rows, remove_high_missing_columns, build_preprocessor
from src.models.explain import run_full_explainability

logger = get_logger(__name__)


def main():
    with open(get_config_path()) as f:
        config = yaml.safe_load(f)

    version = config["artifacts"]["version"]
    model_path     = get_path("models", version, config["artifacts"]["pipeline_filename"])
    preproc_path   = get_path("models", version, "preprocessor.pkl")

    if not model_path.exists():
        logger.error(f"Model not found at {model_path}. Run `python src/models/train.py` first.")
        sys.exit(1)

    logger.info("Loading trained model...")
    model = joblib.load(model_path)
    preprocessor = joblib.load(preproc_path) if preproc_path.exists() else None

    logger.info("Loading master features from cache...")
    master = load_or_build_master()
    master = remove_invalid_rows(master)
    # Do NOT call remove_high_missing_columns here — the saved preprocessor
    # was fitted on all columns (imputer handles NaN internally)

    X = master.drop(columns=["TARGET", "SK_ID_CURR"], errors="ignore")
    y = master["TARGET"]

    _, X_test, _, _ = train_test_split(
        X, y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
        stratify=y,
    )
    X_train, _, _, _ = train_test_split(
        X, y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
        stratify=y,
    )

    if preprocessor is not None:
        import numpy as np
        X_train_proc = pd.DataFrame(preprocessor.transform(X_train), columns=X_train.columns)
        X_test_proc  = pd.DataFrame(preprocessor.transform(X_test),  columns=X_test.columns)
    else:
        X_train_proc = X_train
        X_test_proc  = X_test

    logger.info("Running SHAP analysis...")
    run_full_explainability(model, X_train_proc, X_test_proc)
    logger.info("SHAP analysis complete. Check outputs/plots/")


if __name__ == "__main__":
    main()
