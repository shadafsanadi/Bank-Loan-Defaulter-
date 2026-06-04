"""
Model evaluation utilities.

Provides ROC-AUC, PR-AUC, threshold analysis, calibration check,
and all metrics needed to document final model performance.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    brier_score_loss,
)

from src.utils.logger import get_logger
from src.utils.paths import get_path

logger = get_logger(__name__)


def compute_all_metrics(
    y_true: pd.Series | np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Compute the full evaluation metric suite."""
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "roc_auc":           round(roc_auc_score(y_true, y_proba), 4),
        "pr_auc":            round(average_precision_score(y_true, y_proba), 4),
        "brier_score":       round(brier_score_loss(y_true, y_proba), 4),
        "threshold":         threshold,
        "precision_default": round(tp / (tp + fp + 1e-9), 4),
        "recall_default":    round(tp / (tp + fn + 1e-9), 4),
        "f1_default":        round(2 * tp / (2 * tp + fp + fn + 1e-9), 4),
        "true_positives":    int(tp),
        "false_positives":   int(fp),
        "true_negatives":    int(tn),
        "false_negatives":   int(fn),
        "default_catch_rate":  round(tp / (tp + fn + 1e-9), 4),  # recall alias
        "false_alarm_rate":    round(fp / (fp + tn + 1e-9), 4),
    }

    logger.info(
        f"Metrics at threshold={threshold:.3f}: "
        f"ROC-AUC={metrics['roc_auc']} | PR-AUC={metrics['pr_auc']} | "
        f"Recall={metrics['recall_default']} | Precision={metrics['precision_default']}"
    )
    return metrics


def find_optimal_threshold(
    y_true: pd.Series | np.ndarray,
    y_proba: np.ndarray,
    beta: float = 2.0,
    min_recall: float = 0.70,
) -> float:
    """
    Find threshold maximising F-beta score subject to minimum recall constraint.

    Beta > 1 weights recall more heavily than precision.
    Default beta=2 reflects the assumption that missing a default costs ~4x more
    than wrongly rejecting a good applicant.
    """
    from sklearn.metrics import fbeta_score

    thresholds = np.linspace(0.01, 0.99, 200)
    best_threshold = 0.5
    best_score = -1.0

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        recall = (y_true == 1) & (y_pred == 1)
        recall_rate = recall.sum() / (y_true == 1).sum()

        if recall_rate < min_recall:
            continue

        score = fbeta_score(y_true, y_pred, beta=beta, zero_division=0)
        if score > best_score:
            best_score = score
            best_threshold = t

    logger.info(
        f"Optimal threshold: {best_threshold:.3f} "
        f"(F{beta:.0f}={best_score:.4f}, min_recall={min_recall})"
    )
    return float(best_threshold)


def plot_roc_curve(
    y_true, y_proba, model_name: str, save_dir: Path | None = None
):
    """Plot and optionally save ROC curve."""
    fig, ax = plt.subplots(figsize=(7, 6))
    RocCurveDisplay.from_predictions(y_true, y_proba, name=model_name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", label="Random baseline")
    ax.set_title(f"ROC Curve — {model_name}")
    ax.legend()
    plt.tight_layout()

    if save_dir:
        path = Path(save_dir) / f"roc_curve_{model_name}.png"
        fig.savefig(path, dpi=150)
        logger.info(f"ROC curve saved: {path}")
    return fig


def plot_pr_curve(
    y_true, y_proba, model_name: str, save_dir: Path | None = None
):
    """Plot and optionally save Precision-Recall curve."""
    fig, ax = plt.subplots(figsize=(7, 6))
    PrecisionRecallDisplay.from_predictions(y_true, y_proba, name=model_name, ax=ax)
    baseline = y_true.mean()
    ax.axhline(baseline, color="k", linestyle="--", label=f"Baseline ({baseline:.3f})")
    ax.set_title(f"Precision-Recall Curve — {model_name}")
    ax.legend()
    plt.tight_layout()

    if save_dir:
        path = Path(save_dir) / f"pr_curve_{model_name}.png"
        fig.savefig(path, dpi=150)
        logger.info(f"PR curve saved: {path}")
    return fig


def plot_threshold_analysis(
    y_true, y_proba, save_dir: Path | None = None
):
    """
    Plot precision, recall, and F2 score across probability thresholds.
    Helps the business choose an operating threshold based on their cost model.
    """
    from sklearn.metrics import precision_recall_curve, fbeta_score

    thresholds_grid = np.linspace(0.01, 0.99, 200)
    precisions, recalls, f2_scores = [], [], []

    for t in thresholds_grid:
        y_pred = (y_proba >= t).astype(int)
        prec = y_pred[y_true == 1].sum() / (y_pred.sum() + 1e-9)
        rec  = y_pred[y_true == 1].sum() / (y_true.sum() + 1e-9)
        f2   = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
        precisions.append(prec)
        recalls.append(rec)
        f2_scores.append(f2)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds_grid, precisions, label="Precision", color="blue")
    ax.plot(thresholds_grid, recalls, label="Recall", color="orange")
    ax.plot(thresholds_grid, f2_scores, label="F2 Score", color="green", linewidth=2)
    ax.axvline(0.5, color="red", linestyle="--", alpha=0.5, label="Default threshold (0.5)")
    ax.set_xlabel("Probability Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold Analysis — Precision / Recall / F2")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_dir:
        path = Path(save_dir) / "threshold_analysis.png"
        fig.savefig(path, dpi=150)
        logger.info(f"Threshold analysis saved: {path}")
    return fig


def full_evaluation(
    y_true, y_proba, model_name: str, beta: float = 2.0, min_recall: float = 0.70
) -> dict:
    """Run complete evaluation suite and save all plots."""
    plots_dir = get_path("outputs", "plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    optimal_threshold = find_optimal_threshold(y_true, y_proba, beta=beta, min_recall=min_recall)
    metrics = compute_all_metrics(y_true, y_proba, threshold=optimal_threshold)
    metrics["optimal_threshold"] = optimal_threshold

    plot_roc_curve(y_true, y_proba, model_name, save_dir=plots_dir)
    plot_pr_curve(y_true, y_proba, model_name, save_dir=plots_dir)
    plot_threshold_analysis(y_true, y_proba, save_dir=plots_dir)

    # Print classification report at optimal threshold
    y_pred = (y_proba >= optimal_threshold).astype(int)
    report = classification_report(y_true, y_pred, target_names=["Repay", "Default"])
    logger.info(f"\nClassification report at threshold {optimal_threshold:.3f}:\n{report}")

    return metrics
