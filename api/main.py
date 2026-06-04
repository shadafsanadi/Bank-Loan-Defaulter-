"""
FastAPI prediction service.

Endpoints:
  POST /predict   — predict default risk for a single applicant
  GET  /health    — model version and AUC
  GET  /docs      — auto-generated Swagger UI (built in)

Design decisions:
  - Model artifacts are loaded once at startup (not per-request) via lifespan context
  - Input validation is handled by Pydantic v2 schemas before reaching the model
  - SHAP explanations are included in every prediction response
  - All errors return structured JSON (not plain text)
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.predict import predict_default_risk, get_shap_explanation, get_model_info
from api.schemas import LoanApplication, PredictionResponse, HealthResponse, RiskFactor

# ── Risk band logic (mirrors app.py) ─────────────────────────────────────────
_RISK_BANDS = [
    (0.0,  0.10, "Very Low",  "Applicant shows very strong repayment indicators."),
    (0.10, 0.25, "Low",       "Applicant shows good repayment indicators."),
    (0.25, 0.45, "Medium",    "Moderate risk. Manual review recommended."),
    (0.45, 0.65, "High",      "Elevated risk. Senior review required."),
    (0.65, 1.01, "Very High", "Very high default risk."),
]


def _get_risk_band(prob: float) -> tuple[str, str]:
    for lo, hi, label, desc in _RISK_BANDS:
        if lo <= prob < hi:
            return label, desc
    return "Very High", "Very high default risk."


# ── App with lifespan (model loaded once at startup) ─────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up: verify model artifacts are loadable before accepting requests."""
    try:
        get_model_info()
    except Exception as e:
        print(f"WARNING: Model not loaded — {e}")
        print("Run `python src/models/train.py` to train and save the model.")
    yield


app = FastAPI(
    title="Home Loan Default Prediction API",
    description=(
        "Predicts the probability of loan default using XGBoost trained on "
        "the Home Credit Default Risk dataset. Includes SHAP-based explanations "
        "for every prediction.\n\n"
        "**Model:** XGBoost with 5-fold CV | **Dataset:** Home Credit Default Risk\n\n"
        "**Source code:** [GitHub](https://github.com/Shadaf/HomeLoanDefault)"
    ),
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _application_to_feature_dict(app_data: LoanApplication) -> dict:
    """Convert validated Pydantic schema to feature dict expected by predict_default_risk."""
    d = app_data.model_dump()
    features = {
        "AMT_INCOME_TOTAL":     d["amt_income_total"],
        "AMT_CREDIT":           d["amt_credit"],
        "AMT_ANNUITY":          d["amt_annuity"],
        "DAYS_BIRTH":           -int(d["age_years"] * 365),
        "DAYS_EMPLOYED":        -int(d["employed_years"] * 365) if d["employed_years"] > 0 else 365243,
        "EXT_SOURCE_2":         d["ext_source_2"],
        "EXT_SOURCE_3":         d["ext_source_3"],
        "CODE_GENDER_M":        1 if d["code_gender"] == "M" else 0,
        "CREDIT_INCOME_RATIO":  d["amt_credit"] / (d["amt_income_total"] + 1),
        "ANNUITY_INCOME_RATIO": d["amt_annuity"] / (d["amt_income_total"] + 1),
        "CREDIT_TERM_MONTHS":   d["amt_credit"] / (d["amt_annuity"] + 1),
        "AGE_YEARS":            d["age_years"],
        "EMPLOYED_YEARS":       d["employed_years"],
        "IS_UNEMPLOYED":        1 if d["employed_years"] == 0 else 0,
        "EMPLOYMENT_AGE_RATIO": d["employed_years"] / (d["age_years"] + 1),
        "EXT_SOURCE_PRODUCT":   d["ext_source_2"] * d["ext_source_3"],
        "EXT_SOURCE_MEAN":      (d["ext_source_2"] + d["ext_source_3"]) / 2,
    }

    # Categorical one-hot flags
    income_type_map = {
        "Pensioner":    "NAME_INCOME_TYPE_Pensioner",
        "State servant":"NAME_INCOME_TYPE_State servant",
        "Unemployed":   "NAME_INCOME_TYPE_Unemployed",
        "Working":      "NAME_INCOME_TYPE_Working",
    }
    if d["name_income_type"] in income_type_map:
        features[income_type_map[d["name_income_type"]]] = 1

    edu_map = {
        "Higher education":              "NAME_EDUCATION_TYPE_Higher education",
        "Incomplete higher":             "NAME_EDUCATION_TYPE_Incomplete higher",
        "Lower secondary":               "NAME_EDUCATION_TYPE_Lower secondary",
        "Secondary / secondary special": "NAME_EDUCATION_TYPE_Secondary / secondary special",
    }
    if d["name_education_type"] in edu_map:
        features[edu_map[d["name_education_type"]]] = 1

    family_map = {
        "Married":              "NAME_FAMILY_STATUS_Married",
        "Separated":            "NAME_FAMILY_STATUS_Separated",
        "Single / not married": "NAME_FAMILY_STATUS_Single / not married",
        "Widow":                "NAME_FAMILY_STATUS_Widow",
    }
    if d["name_family_status"] in family_map:
        features[family_map[d["name_family_status"]]] = 1

    housing_map = {
        "House / apartment":   "NAME_HOUSING_TYPE_House / apartment",
        "Municipal apartment": "NAME_HOUSING_TYPE_Municipal apartment",
        "Office apartment":    "NAME_HOUSING_TYPE_Office apartment",
        "Rented apartment":    "NAME_HOUSING_TYPE_Rented apartment",
        "With parents":        "NAME_HOUSING_TYPE_With parents",
    }
    if d["name_housing_type"] in housing_map:
        features[housing_map[d["name_housing_type"]]] = 1

    return features


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict loan default risk",
    description=(
        "Returns a default probability, risk band, and SHAP-based explanation "
        "of the top factors driving the prediction."
    ),
)
async def predict(application: LoanApplication) -> PredictionResponse:
    try:
        feature_dict = _application_to_feature_dict(application)
        prediction, probability = predict_default_risk(feature_dict)
        risk_band, risk_desc = _get_risk_band(probability)

        shap_factors = get_shap_explanation(feature_dict) or []
        risk_factors = [
            RiskFactor(
                feature=f["feature"],
                shap_value=f["shap_value"],
                direction="increases_risk" if f["shap_value"] > 0 else "decreases_risk",
            )
            for f in shap_factors[:5]
        ]

        return PredictionResponse(
            prediction=prediction,
            probability=round(probability, 4),
            risk_band=risk_band,
            risk_description=risk_desc,
            top_risk_factors=risk_factors,
        )

    except (FileNotFoundError, RuntimeError) as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Model health check",
)
async def health() -> HealthResponse:
    try:
        info = get_model_info()
        return HealthResponse(
            status="ok",
            model_name=info.get("model_name", "unknown"),
            model_version="v2",
            test_roc_auc=info.get("test_roc_auc"),
            trained_at=info.get("trained_at"),
            n_features=info.get("n_features"),
        )
    except Exception as e:
        return HealthResponse(
            status="degraded",
            model_name="unknown",
            model_version="unknown",
            test_roc_auc=None,
            trained_at=None,
            n_features=None,
        )


@app.get("/", include_in_schema=False)
async def root():
    return JSONResponse({"message": "Home Loan Default Prediction API", "docs": "/docs"})
