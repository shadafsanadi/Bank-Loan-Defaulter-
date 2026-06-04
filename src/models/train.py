"""
Training pipeline with 5-fold stratified cross-validation and multi-model comparison.

Execution flow:
  1. Load master features (from cache or rebuilt from raw files)
  2. Split into development set (85%) and held-out test set (15%)
  3. Run CV on development set to compare all models
  4. Select best model, retrain on full development set
  5. Evaluate once on test set and save results
  6. Save pipeline artifact + metadata
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier

# ── Optional imports ──────────────────────────────────────────────────────────
try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.paths import get_path, get_config_path
from src.utils.logger import get_logger
from src.features.pipeline import load_or_build_master
from src.preprocessing import build_preprocessor

logger = get_logger(__name__)


def load_config() -> dict:
    with open(get_config_path()) as f:
        return yaml.safe_load(f)


def prepare_data(config: dict) -> tuple:
    """Load master features and return X_dev, X_test, y_dev, y_test."""
    master = load_or_build_master()

    # Remove invalid rows
    if "CODE_GENDER" in master.columns:
        master = master[master["CODE_GENDER"] != "XNA"].reset_index(drop=True)

    X = master.drop(columns=["TARGET", "SK_ID_CURR"], errors="ignore")
    y = master["TARGET"]

    return train_test_split(
        X, y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
        stratify=y,
    )


def get_models(config: dict) -> dict:
    """Instantiate all models from config."""
    spw = config["scale_pos_weight"]
    rs  = config["data"]["random_state"]

    models = {
        "logistic_regression": LogisticRegression(
            class_weight="balanced",
            C=config["logistic_regression"]["C"],
            max_iter=config["logistic_regression"]["max_iter"],
            random_state=rs,
            n_jobs=-1,
        ),
        "random_forest": RandomForestClassifier(
            class_weight="balanced",
            n_estimators=config["random_forest"]["n_estimators"],
            max_depth=config["random_forest"]["max_depth"],
            min_samples_leaf=config["random_forest"]["min_samples_leaf"],
            random_state=rs,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=config["xgboost"]["n_estimators"],
            max_depth=config["xgboost"]["max_depth"],
            learning_rate=config["xgboost"]["learning_rate"],
            subsample=config["xgboost"]["subsample"],
            colsample_bytree=config["xgboost"]["colsample_bytree"],
            min_child_weight=config["xgboost"]["min_child_weight"],
            gamma=config["xgboost"]["gamma"],
            reg_alpha=config["xgboost"]["reg_alpha"],
            reg_lambda=config["xgboost"]["reg_lambda"],
            scale_pos_weight=spw,
            eval_metric=config["xgboost"]["eval_metric"],
            tree_method=config["xgboost"]["tree_method"],
            n_jobs=config["xgboost"]["n_jobs"],
            random_state=rs,
        ),
    }

    if HAS_LGBM:
        models["lightgbm"] = lgb.LGBMClassifier(
            n_estimators=config["lightgbm"]["n_estimators"],
            learning_rate=config["lightgbm"]["learning_rate"],
            num_leaves=config["lightgbm"]["num_leaves"],
            min_child_samples=config["lightgbm"]["min_child_samples"],
            subsample=config["lightgbm"]["subsample"],
            colsample_bytree=config["lightgbm"]["colsample_bytree"],
            reg_alpha=config["lightgbm"]["reg_alpha"],
            reg_lambda=config["lightgbm"]["reg_lambda"],
            scale_pos_weight=spw,
            n_jobs=config["lightgbm"]["n_jobs"],
            verbose=config["lightgbm"]["verbose"],
            random_state=rs,
        )
    else:
        logger.warning("LightGBM not installed — skipping. Run: pip install lightgbm")

    if HAS_CATBOOST:
        models["catboost"] = CatBoostClassifier(
            iterations=config["catboost"]["iterations"],
            learning_rate=config["catboost"]["learning_rate"],
            depth=config["catboost"]["depth"],
            l2_leaf_reg=config["catboost"]["l2_leaf_reg"],
            scale_pos_weight=spw,
            verbose=config["catboost"]["verbose"],
            random_state=rs,
        )
    else:
        logger.warning("CatBoost not installed — skipping. Run: pip install catboost")

    return models


def run_cross_validation(
    X: pd.DataFrame,
    y: pd.Series,
    models: dict,
    config: dict,
) -> pd.DataFrame:
    """Run 5-fold stratified CV for all models. Returns comparison DataFrame."""
    cv = StratifiedKFold(
        n_splits=config["cv"]["n_splits"],
        shuffle=config["cv"]["shuffle"],
        random_state=config["cv"]["random_state"],
    )

    results = []
    for name, model in models.items():
        logger.info(f"Cross-validating: {name}")

        # Handle XGBoost early stopping — requires eval_set which cross_val_score
        # doesn't support natively; disable for CV, enable for final fit
        if hasattr(model, "early_stopping_rounds"):
            model = model.set_params(early_stopping_rounds=None)

        scores = cross_val_score(
            model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1, verbose=0
        )
        results.append({
            "model":        name,
            "cv_auc_mean":  round(scores.mean(), 4),
            "cv_auc_std":   round(scores.std(), 4),
            "cv_auc_min":   round(scores.min(), 4),
            "cv_auc_max":   round(scores.max(), 4),
        })
        logger.info(
            f"  {name}: {scores.mean():.4f} ± {scores.std():.4f} "
            f"(min={scores.min():.4f}, max={scores.max():.4f})"
        )

    comparison = pd.DataFrame(results).sort_values("cv_auc_mean", ascending=False)
    logger.info("\nModel comparison:\n" + comparison.to_string(index=False))
    return comparison


def train_final_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    models: dict,
    best_model_name: str,
    config: dict,
) -> object:
    """Train the best model on the full development set with early stopping."""
    model = models[best_model_name]

    if best_model_name == "xgboost":
        model = model.set_params(
            early_stopping_rounds=config["xgboost"]["early_stopping_rounds"]
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=100,
        )
    elif best_model_name == "lightgbm" and HAS_LGBM:
        model = model.set_params(
            early_stopping_rounds=config["lightgbm"]["early_stopping_rounds"]
        )
        callbacks = [lgb.early_stopping(config["lightgbm"]["early_stopping_rounds"]),
                     lgb.log_evaluation(100)]
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=callbacks,
        )
    elif best_model_name == "catboost" and HAS_CATBOOST:
        model.fit(X_train, y_train, eval_set=(X_val, y_val))
    else:
        model.fit(X_train, y_train)

    return model


def save_artifacts(
    model,
    feature_names: list[str],
    metadata: dict,
    config: dict,
) -> Path:
    """Save model pipeline, feature names, and metadata."""
    version = config["artifacts"]["version"]
    models_dir = get_path(config["artifacts"]["models_dir"], version)
    models_dir.mkdir(parents=True, exist_ok=True)

    pipeline_path = models_dir / config["artifacts"]["pipeline_filename"]
    features_path = models_dir / config["artifacts"]["feature_names_filename"]
    metadata_path = models_dir / config["artifacts"]["metadata_filename"]

    joblib.dump(model, pipeline_path)
    joblib.dump(feature_names, features_path)

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Artifacts saved to {models_dir}")
    return models_dir


def train():
    logger.info("=" * 60)
    logger.info("Home Loan Default Prediction — Training Pipeline")
    logger.info("=" * 60)

    config = load_config()

    # ── Data preparation ──────────────────────────────────────────────────
    logger.info("Preparing data...")
    X_dev, X_test, y_dev, y_test = prepare_data(config)
    logger.info(f"Dev: {len(X_dev):,} rows | Test: {len(X_test):,} rows (locked)")
    logger.info(f"Default rate — Dev: {y_dev.mean():.3%} | Test: {y_test.mean():.3%}")

    # ── Preprocessing ─────────────────────────────────────────────────────
    preprocessor = build_preprocessor(X_dev)
    X_dev_proc  = preprocessor.fit_transform(X_dev)
    X_test_proc = preprocessor.transform(X_test)

    # ── Model comparison ──────────────────────────────────────────────────
    models = get_models(config)
    logger.info(f"Comparing {len(models)} models with {config['cv']['n_splits']}-fold CV...")
    comparison = run_cross_validation(X_dev_proc, y_dev, models, config)

    # Save comparison results
    out_dir = get_path("outputs")
    out_dir.mkdir(exist_ok=True)
    comparison.to_csv(out_dir / "model_comparison.csv", index=False)

    # ── Final training ────────────────────────────────────────────────────
    best_name = comparison.iloc[0]["model"]
    logger.info(f"\nBest model: {best_name} (CV AUC = {comparison.iloc[0]['cv_auc_mean']})")

    # Use 90% of dev for final training, 10% as early-stopping validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev_proc, y_dev, test_size=0.1,
        random_state=config["data"]["random_state"],
        stratify=y_dev,
    )

    logger.info(f"Training final {best_name} model...")
    final_model = train_final_model(
        X_train, y_train, X_val, y_val, models, best_name, config
    )

    # ── Test set evaluation (run ONCE) ────────────────────────────────────
    logger.info("\n--- HELD-OUT TEST SET EVALUATION ---")
    y_proba = final_model.predict_proba(X_test_proc)[:, 1]
    test_auc = roc_auc_score(y_test, y_proba)
    test_pr_auc = average_precision_score(y_test, y_proba)
    logger.info(f"Test ROC-AUC : {test_auc:.4f}")
    logger.info(f"Test PR-AUC  : {test_pr_auc:.4f}")

    # ── Save artifacts ────────────────────────────────────────────────────
    metadata = {
        "model_name":        best_name,
        "trained_at":        datetime.now().isoformat(),
        "test_roc_auc":      round(test_auc, 4),
        "test_pr_auc":       round(test_pr_auc, 4),
        "cv_roc_auc_mean":   float(comparison.iloc[0]["cv_auc_mean"]),
        "cv_roc_auc_std":    float(comparison.iloc[0]["cv_auc_std"]),
        "n_features":        X_dev_proc.shape[1],
        "n_train_rows":      len(X_dev),
        "default_rate":      round(float(y_dev.mean()), 4),
        "model_comparison":  comparison.to_dict("records"),
    }

    save_artifacts(final_model, list(X_dev.columns), metadata, config)

    # Also save the preprocessor separately for inference
    version = config["artifacts"]["version"]
    preproc_path = get_path(config["artifacts"]["models_dir"], version, "preprocessor.pkl")
    joblib.dump(preprocessor, preproc_path)

    logger.info("\nTraining complete.")
    logger.info(f"Final model: {best_name}")
    logger.info(f"Test ROC-AUC: {test_auc:.4f} | PR-AUC: {test_pr_auc:.4f}")


if __name__ == "__main__":
    train()
