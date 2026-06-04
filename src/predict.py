"""
Inference module — loads trained artifacts and serves predictions.

Supports two artifact versions:
  v1: Legacy MVP (separate xgb_model.pkl + scaler.pkl)
  v2: Current (pipeline.pkl + preprocessor.pkl + shap_explainer.pkl)

The predict_default_risk() function is the single public interface used by
both the Streamlit app and the FastAPI service.
"""

import json
import os
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.paths import get_path, get_config_path
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _load_config() -> dict:
    with open(get_config_path()) as f:
        return yaml.safe_load(f)


def _load_artifacts() -> tuple:
    """
    Load model + preprocessor + feature names from the active artifact version.
    Falls back to v1 if v2 artifacts are not found.
    """
    config = _load_config()
    version = config["artifacts"]["version"]
    models_dir = get_path("models", version)

    pipeline_path  = models_dir / config["artifacts"]["pipeline_filename"]
    features_path  = models_dir / config["artifacts"]["feature_names_filename"]
    metadata_path  = models_dir / config["artifacts"]["metadata_filename"]
    preproc_path   = models_dir / "preprocessor.pkl"
    explainer_path = models_dir / "shap_explainer.pkl"

    # ── v2 artifacts ──────────────────────────────────────────────────────
    if pipeline_path.exists() and features_path.exists():
        model         = joblib.load(pipeline_path)
        feature_names = joblib.load(features_path)
        preprocessor  = joblib.load(preproc_path) if preproc_path.exists() else None
        explainer     = joblib.load(explainer_path) if explainer_path.exists() else None
        threshold     = 0.5
        if metadata_path.exists():
            with open(metadata_path) as f:
                meta = json.load(f)
            threshold = meta.get("optimal_threshold", 0.5)
        logger.info(f"Loaded v2 artifacts from {models_dir}")
        return model, feature_names, preprocessor, explainer, threshold

    # ── v1 fallback ───────────────────────────────────────────────────────
    v1_dir = get_path("models")
    v1_model_path    = v1_dir / "xgb_model.pkl"
    v1_features_path = v1_dir / "model_features.pkl"
    v1_defaults_path = v1_dir / "feature_defaults.pkl"

    if v1_model_path.exists():
        logger.warning("v2 artifacts not found — falling back to v1 (MVP) model.")
        model         = joblib.load(v1_model_path)
        feature_names = joblib.load(v1_features_path) if v1_features_path.exists() else []
        defaults      = joblib.load(v1_defaults_path) if v1_defaults_path.exists() else {}
        return model, feature_names, None, None, 0.5

    raise FileNotFoundError(
        "No model artifacts found. Run `python src/models/train.py` to train the model."
    )


# Load artifacts once at module import time (not per-request)
try:
    _model, _feature_names, _preprocessor, _explainer, _threshold = _load_artifacts()
    _artifacts_loaded = True
except FileNotFoundError as e:
    logger.warning(str(e))
    _artifacts_loaded = False
    _model = _feature_names = _preprocessor = _explainer = _threshold = None


def predict_default_risk(input_dict: dict) -> tuple[int, float]:
    """
    Predict loan default risk from a feature dictionary.

    Args:
        input_dict: {feature_name: value} for any subset of model features.
                    Unrecognised features are ignored.
                    Missing features default to 0 (appropriate for behavioral features).

    Returns:
        (prediction, probability)
          prediction: 0 = low risk, 1 = high risk
          probability: float in [0, 1] — probability of default
    """
    if not _artifacts_loaded:
        raise RuntimeError(
            "Model artifacts not loaded. Run `python src/models/train.py` first."
        )

    unknown = [k for k in input_dict if k not in _feature_names]
    if unknown:
        warnings.warn(f"Ignoring unrecognised features: {unknown}", stacklevel=2)

    # Build input row with zeros for missing features (not median — see FEATURE_ENGINEERING.md)
    row = {f: input_dict.get(f, 0) for f in _feature_names}
    input_df = pd.DataFrame([row], columns=_feature_names)

    if _preprocessor is not None:
        input_proc = _preprocessor.transform(input_df)
        probability = float(_model.predict_proba(input_proc)[0][1])
    else:
        probability = float(_model.predict_proba(input_df)[0][1])

    prediction = int(probability >= _threshold)
    return prediction, probability


def get_shap_explanation(input_dict: dict) -> list[dict] | None:
    """
    Return top SHAP feature contributions for a single prediction.
    Returns None if SHAP explainer is not available.
    """
    if _explainer is None or not _artifacts_loaded:
        return None

    row = {f: input_dict.get(f, 0) for f in _feature_names}
    input_df = pd.DataFrame([row], columns=_feature_names)

    if _preprocessor is not None:
        input_proc = _preprocessor.transform(input_df)
        input_for_shap = pd.DataFrame(input_proc, columns=_feature_names)
    else:
        input_for_shap = input_df

    shap_vals = _explainer.shap_values(input_for_shap)[0]
    factors = sorted(
        [{"feature": f, "shap_value": round(float(v), 4)}
         for f, v in zip(_feature_names, shap_vals)],
        key=lambda x: abs(x["shap_value"]),
        reverse=True,
    )
    return factors[:10]


def get_model_info() -> dict:
    """Return metadata about the loaded model (used by /health endpoint)."""
    if not _artifacts_loaded:
        return {"status": "no model loaded"}

    config = _load_config()
    version = config["artifacts"]["version"]
    metadata_path = get_path("models", version, config["artifacts"]["metadata_filename"])

    if metadata_path.exists():
        with open(metadata_path) as f:
            return json.load(f)
    return {"model_version": version, "status": "metadata not found"}
