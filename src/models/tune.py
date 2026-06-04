"""
Hyperparameter tuning with Optuna.

Uses TPE (Tree-structured Parzen Estimator) sampler which outperforms
grid/random search by modelling the objective function and sampling promising
regions of the parameter space.

Pruning: MedianPruner stops unpromising trials early using CV fold scores,
saving significant compute time.

Run after initial model comparison identifies the best algorithm.
Usage:
    python src/models/tune.py --model xgboost --n-trials 100
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import optuna
import yaml
from sklearn.model_selection import StratifiedKFold, cross_val_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.paths import get_path, get_config_path
from src.utils.logger import get_logger
from src.features.pipeline import load_or_build_master
from src.preprocessing import build_preprocessor, remove_high_missing_columns, remove_invalid_rows

logger = get_logger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_data():
    master = load_or_build_master()
    master = remove_invalid_rows(master)
    master = remove_high_missing_columns(master)
    X = master.drop(columns=["TARGET", "SK_ID_CURR"], errors="ignore")
    y = master["TARGET"]
    preprocessor = build_preprocessor(X)
    X_proc = preprocessor.fit_transform(X)
    return X_proc, y, preprocessor


def _xgboost_objective(trial, X, y, cv, config):
    from xgboost import XGBClassifier

    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 200, 1200),
        "max_depth":        trial.suggest_int("max_depth", 3, 9),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "gamma":            trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "scale_pos_weight": config["scale_pos_weight"],
        "tree_method":      "hist",
        "eval_metric":      "auc",
        "n_jobs":           -1,
        "random_state":     42,
        "early_stopping_rounds": None,  # disabled for CV
    }

    model = XGBClassifier(**params)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=1)
    return scores.mean()


def _lightgbm_objective(trial, X, y, cv, config):
    import lightgbm as lgb

    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 200, 1200),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves":       trial.suggest_int("num_leaves", 20, 300),
        "max_depth":        trial.suggest_int("max_depth", 3, 12),
        "min_child_samples":trial.suggest_int("min_child_samples", 20, 200),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "scale_pos_weight": config["scale_pos_weight"],
        "n_jobs":           -1,
        "verbose":          -1,
        "random_state":     42,
    }

    model = lgb.LGBMClassifier(**params)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=1)
    return scores.mean()


_OBJECTIVES = {
    "xgboost":  _xgboost_objective,
    "lightgbm": _lightgbm_objective,
}


def tune(model_name: str = "xgboost", n_trials: int = 50) -> dict:
    """
    Run Optuna hyperparameter search for the specified model.

    Args:
        model_name: "xgboost" or "lightgbm"
        n_trials:   Number of Optuna trials

    Returns:
        Best hyperparameters found
    """
    if model_name not in _OBJECTIVES:
        raise ValueError(f"Unsupported model: {model_name}. Choose from {list(_OBJECTIVES)}")

    with open(get_config_path()) as f:
        config = yaml.safe_load(f)

    logger.info(f"Loading data for tuning...")
    X, y, _ = load_data()

    cv = StratifiedKFold(
        n_splits=config["cv"]["n_splits"],
        shuffle=True,
        random_state=config["cv"]["random_state"],
    )

    objective_fn = _OBJECTIVES[model_name]

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5),
    )

    logger.info(f"Starting Optuna search: {n_trials} trials for {model_name}...")

    study.optimize(
        lambda trial: objective_fn(trial, X, y, cv, config),
        n_trials=n_trials,
        timeout=config["tuning"]["timeout_seconds"],
        show_progress_bar=True,
    )

    best_params = study.best_params
    best_auc    = study.best_value

    logger.info(f"\nBest {model_name} AUC: {best_auc:.4f}")
    logger.info(f"Best params: {best_params}")

    # Save results
    out_dir = get_path("outputs")
    out_dir.mkdir(exist_ok=True)
    results_path = out_dir / f"optuna_{model_name}_results.csv"
    study.trials_dataframe().to_csv(results_path, index=False)
    logger.info(f"Trial results saved to {results_path}")

    return best_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter tuning with Optuna")
    parser.add_argument("--model",    default="xgboost", choices=list(_OBJECTIVES))
    parser.add_argument("--n-trials", type=int, default=50)
    args = parser.parse_args()
    tune(args.model, args.n_trials)
