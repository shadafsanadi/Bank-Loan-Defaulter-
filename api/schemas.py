"""
Pydantic v2 request/response schemas for the FastAPI prediction service.

All fields have realistic min/max constraints derived from the training dataset's
95th-percentile ranges. Out-of-range inputs are rejected with a 422 error before
reaching the model, preventing silent garbage-in predictions.
"""

from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field


class LoanApplication(BaseModel):
    """Prediction request — core applicant fields."""

    # ── Financials ─────────────────────────────────────────────────────────
    amt_income_total: float = Field(
        ..., ge=10_000, le=100_000_000,
        description="Annual income in local currency",
        example=150_000,
    )
    amt_credit: float = Field(
        ..., ge=10_000, le=10_000_000,
        description="Requested loan amount",
        example=500_000,
    )
    amt_annuity: float = Field(
        ..., ge=1_000, le=1_000_000,
        description="Annual loan repayment amount",
        example=25_000,
    )

    # ── External credit scores ─────────────────────────────────────────────
    ext_source_2: float = Field(
        0.5, ge=0.0, le=1.0,
        description="Normalised external credit score 2 (0=worst, 1=best)",
        example=0.65,
    )
    ext_source_3: float = Field(
        0.5, ge=0.0, le=1.0,
        description="Normalised external credit score 3 (0=worst, 1=best)",
        example=0.52,
    )

    # ── Demographics ──────────────────────────────────────────────────────
    age_years: int = Field(
        ..., ge=18, le=100,
        description="Applicant age in years",
        example=35,
    )
    employed_years: float = Field(
        0.0, ge=0.0, le=60.0,
        description="Years at current employer. 0 = unemployed.",
        example=5.0,
    )
    code_gender: Literal["F", "M"] = Field(
        "F",
        description="Applicant gender",
        example="F",
    )

    # ── Categorical fields ─────────────────────────────────────────────────
    name_income_type: Literal[
        "Commercial associate", "Pensioner", "State servant", "Unemployed", "Working"
    ] = Field("Working", example="Working")

    name_education_type: Literal[
        "Academic degree", "Higher education", "Incomplete higher",
        "Lower secondary", "Secondary / secondary special",
    ] = Field("Higher education", example="Higher education")

    name_family_status: Literal[
        "Civil marriage", "Married", "Separated", "Single / not married", "Widow"
    ] = Field("Married", example="Married")

    name_housing_type: Literal[
        "Co-op apartment", "House / apartment", "Municipal apartment",
        "Office apartment", "Rented apartment", "With parents",
    ] = Field("House / apartment", example="House / apartment")

    class Config:
        json_schema_extra = {
            "example": {
                "amt_income_total": 150_000,
                "amt_credit": 500_000,
                "amt_annuity": 25_000,
                "ext_source_2": 0.65,
                "ext_source_3": 0.52,
                "age_years": 35,
                "employed_years": 5.0,
                "code_gender": "F",
                "name_income_type": "Working",
                "name_education_type": "Higher education",
                "name_family_status": "Married",
                "name_housing_type": "House / apartment",
            }
        }


class RiskFactor(BaseModel):
    feature: str
    shap_value: float
    direction: Literal["increases_risk", "decreases_risk"]


class PredictionResponse(BaseModel):
    prediction: Literal[0, 1]
    probability: float = Field(..., ge=0.0, le=1.0)
    risk_band: Literal["Very Low", "Low", "Medium", "High", "Very High"]
    risk_description: str
    top_risk_factors: list[RiskFactor]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_name: str
    model_version: str
    test_roc_auc: float | None
    trained_at: str | None
    n_features: int | None
