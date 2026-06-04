"""
Model explainability using SHAP TreeExplainer.

Provides:
  - Global feature importance (mean |SHAP value| across test set)
  - Per-prediction waterfall plot (for Streamlit app)
  - Dependence plots for top features
  - SHAP summary plot

SHAP (SHapley Additive exPlanations) is theoretically grounded in cooperative
game theory: each feature's SHAP value represents its marginal contribution
to the prediction, averaged over all possible feature orderings.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from src.utils.logger import get_logger
from src.utils.paths import get_path

logger = get_logger(__name__)


def build_explainer(model, X_background: pd.DataFrame) -> shap.TreeExplainer:
    """
    Build a SHAP TreeExplainer for tree-based models.

    TreeExplainer uses the model's tree structure directly — exact SHAP values
    computed in O(TLD) where T=trees, L=leaves, D=depth. Much faster than
    KernelExplainer.

    Args:
        model: Trained XGBoost / LightGBM / CatBoost model
        X_background: A sample of training data (used to estimate expected value)
    """
    logger.info("Building SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(
        model,
        data=shap.sample(X_background, 100),  # 100-row background = stable expected value
        feature_perturbation="interventional",
    )
    return explainer


def compute_shap_values(
    explainer: shap.TreeExplainer,
    X: pd.DataFrame,
    max_rows: int = 5000,
) -> np.ndarray:
    """
    Compute SHAP values for a dataset.

    Limits to max_rows to keep computation time reasonable for the global summary.
    Full predictions use the per-row method in explain_single_prediction().
    """
    sample = X.iloc[:max_rows] if len(X) > max_rows else X
    logger.info(f"Computing SHAP values for {len(sample):,} rows...")
    shap_values = explainer.shap_values(sample)
    logger.info("SHAP values computed.")
    return shap_values, sample


def plot_global_importance(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    top_n: int = 20,
    save_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Plot global feature importance ranked by mean |SHAP value|.

    Returns a DataFrame of feature importances for further analysis.
    """
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": X_sample.columns,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)

    # Bar chart
    fig, ax = plt.subplots(figsize=(10, 8))
    top = importance_df.head(top_n)
    ax.barh(top["feature"][::-1], top["mean_abs_shap"][::-1], color="#2196F3")
    ax.set_xlabel("Mean |SHAP Value|")
    ax.set_title(f"Top {top_n} Features by Global SHAP Importance")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()

    if save_dir:
        path = Path(save_dir) / "shap_global_importance.png"
        fig.savefig(path, dpi=150)
        logger.info(f"Global importance plot saved: {path}")

    return importance_df


def plot_shap_summary(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    save_dir: Path | None = None,
):
    """
    SHAP summary plot (beeswarm): shows direction and magnitude of each feature's impact.

    Red = high feature value, Blue = low feature value.
    Position on x-axis = SHAP value (impact on model output).
    """
    fig = plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_sample, show=False, max_display=25)
    plt.tight_layout()

    if save_dir:
        path = Path(save_dir) / "shap_summary.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"SHAP summary plot saved: {path}")

    return fig


def explain_single_prediction(
    explainer: shap.TreeExplainer,
    row: pd.DataFrame,
    feature_names: list[str] | None = None,
) -> tuple[shap.Explanation, float]:
    """
    Compute SHAP explanation for a single prediction row.

    Returns:
        explanation: shap.Explanation object (use for waterfall plot)
        base_value: model's expected output (prior to feature contributions)
    """
    explanation = explainer(row)
    return explanation, explainer.expected_value


def get_top_risk_factors(
    explainer: shap.TreeExplainer,
    row: pd.DataFrame,
    n: int = 5,
) -> list[dict]:
    """
    Return the top n features driving a prediction (for API response).

    Positive SHAP = increases default probability
    Negative SHAP = decreases default probability
    """
    shap_vals = explainer.shap_values(row)[0]
    feature_names = row.columns.tolist()

    factors = sorted(
        [{"feature": f, "shap_value": round(float(v), 4)}
         for f, v in zip(feature_names, shap_vals)],
        key=lambda x: abs(x["shap_value"]),
        reverse=True,
    )
    return factors[:n]


def run_full_explainability(model, X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Run complete SHAP analysis and save all plots."""
    plots_dir = get_path("outputs", "plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    explainer = build_explainer(model, X_train)
    shap_values, X_sample = compute_shap_values(explainer, X_test, max_rows=3000)

    importance_df = plot_global_importance(shap_values, X_sample, save_dir=plots_dir)
    plot_shap_summary(shap_values, X_sample, save_dir=plots_dir)

    # Save importance CSV
    importance_df.to_csv(get_path("outputs", "feature_importance.csv"), index=False)
    logger.info("SHAP analysis complete. Results saved to outputs/")

    # Save explainer for app use
    import joblib
    from src.utils.paths import get_path as gp
    import yaml
    config_path = gp("configs", "model_config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    version = config["artifacts"]["version"]
    joblib.dump(explainer, gp("models", version, "shap_explainer.pkl"))
    logger.info("SHAP explainer saved to model artifacts.")

    return explainer, importance_df
