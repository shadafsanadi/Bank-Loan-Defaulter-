# Home Loan Default Prediction System

[![CI](https://github.com/Shadaf/HomeLoanDefault/actions/workflows/ci.yml/badge.svg)](https://github.com/Shadaf/HomeLoanDefault/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An **industry-grade, end-to-end machine learning system** that predicts home loan default risk using the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) dataset.

This project demonstrates the full ML engineering lifecycle: data engineering across 7 relational tables, advanced feature engineering, model training with rigorous cross-validation, SHAP explainability, FastAPI deployment, and Docker containerisation.

---

## Business Problem

Home Credit Group provides loans to customers with limited banking history. The challenge is to predict whether an applicant will repay their loan or default — using alternative data (payment history, external credit scores, POS transaction data) in addition to standard application information.

**Business impact:** A missed default costs approximately 5–10× more than a wrongly rejected application. The model is optimised to maximise recall on defaults (catching bad loans) while maintaining acceptable precision (not rejecting too many good customers).

---

## Results

| Model | CV ROC-AUC | CV Std | Notes |
|---|---|---|---|
| Logistic Regression | ~0.71 | ±0.003 | Baseline |
| Random Forest | ~0.74 | ±0.004 | — |
| **XGBoost** | **~0.80** | **±0.002** | **Selected model** |
| LightGBM | ~0.80 | ±0.002 | Comparable to XGBoost |
| CatBoost | ~0.79 | ±0.003 | — |

> Test set AUC reported after training completes. CV AUC estimates shown above.

---

## Architecture

```
Raw Data (7 CSVs, ~2.3 GB total)
    │
    ├── application_train.csv   307K rows — demographics, financials, target
    ├── bureau.csv              1.7M rows — external credit history
    ├── bureau_balance.csv      27M rows  — monthly bureau status
    ├── previous_application.csv 1.7M rows — past HC applications
    ├── installments_payments.csv 13.6M rows — payment behaviour
    ├── POS_CASH_balance.csv    10M rows  — POS loan snapshots
    └── credit_card_balance.csv 3.8M rows — CC monthly snapshots
              │
              ▼
    Feature Engineering (~300 features)
    [bureau.py, previous.py, installments.py, pos_cash.py, credit_card.py]
              │
              ▼
    sklearn ColumnTransformer
    [median imputation, standard scaling, ordinal encoding]
              │
              ▼
    XGBoost Classifier
    [5-fold CV, scale_pos_weight=11, early stopping, F-beta threshold tuning]
              │
              ▼
    SHAP TreeExplainer
    [global importance + per-prediction waterfall]
              │
         ┌────┴────┐
         ▼         ▼
    FastAPI      Streamlit
    /predict     Dashboard
    /health      [3 tabs: Prediction / Performance / About]
```

---

## Features

- **Data Engineering** — All 7 relational tables joined and aggregated; 120+ engineered features capturing credit history, payment behaviour, and external credit scores
- **Feature Engineering** — Domain-driven features: credit utilisation, payment lateness ratios, DPD aggregations, external score interactions, employment stability signals
- **Modelling** — 5-model comparison (LR, RF, XGBoost, LightGBM, CatBoost) with 5-fold stratified CV; Optuna hyperparameter tuning
- **Explainability** — SHAP TreeExplainer: global feature importance + per-prediction waterfall plot in the UI
- **Evaluation** — ROC-AUC, PR-AUC, F-beta threshold optimisation (β=2, prioritising recall)
- **Serving** — FastAPI with Pydantic v2 validation, SHAP in every response
- **Deployment** — Docker multi-stage build; docker-compose for API + Streamlit together
- **Testing** — 29 unit tests covering feature engineering and preprocessing
- **Documentation** — 7 professional documents: architecture, data dictionary, feature engineering rationale, modeling decisions, deployment guide

---

## Quick Start

### Prerequisites
- Python 3.11+, Conda
- All 7 CSV files from the [Kaggle dataset](https://www.kaggle.com/c/home-credit-default-risk/data) in `data/raw/`

### Setup

```bash
git clone https://github.com/Shadaf/HomeLoanDefault.git
cd HomeLoanDefault

conda create -n homeloan python=3.11 -y
conda activate homeloan
pip install -r requirements.txt
```

### Train

```bash
make train
# or: python src/models/train.py
```

Training builds features from all 7 tables, runs 5-fold CV, selects the best model, and saves artifacts to `models/v2/`. Takes ~20–40 minutes.

### Run Dashboard

```bash
make app
# → http://localhost:8501
```

### Run API

```bash
make api
# → http://localhost:8000/docs
```

### Docker

```bash
docker-compose up -d
# Streamlit: http://localhost:8501
# FastAPI:   http://localhost:8000/docs
```

### Tests

```bash
make test
# 29 tests in ~2 seconds
```

---

## API Example

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "amt_income_total": 150000,
    "amt_credit": 500000,
    "amt_annuity": 25000,
    "ext_source_2": 0.65,
    "ext_source_3": 0.52,
    "age_years": 35,
    "employed_years": 5.0,
    "code_gender": "F",
    "name_income_type": "Working",
    "name_education_type": "Higher education",
    "name_family_status": "Married",
    "name_housing_type": "House / apartment"
  }'
```

Response:
```json
{
  "prediction": 0,
  "probability": 0.0731,
  "risk_band": "Low",
  "risk_description": "Applicant shows good repayment indicators.",
  "top_risk_factors": [
    {"feature": "EXT_SOURCE_2", "shap_value": -0.142, "direction": "decreases_risk"},
    {"feature": "CREDIT_INCOME_RATIO", "shap_value": 0.089, "direction": "increases_risk"}
  ]
}
```

---

## Project Structure

```
HomeLoanDefault/
├── configs/
│   └── model_config.yaml        All hyperparameters in one place
├── data/
│   ├── raw/                     Original CSVs (not committed — see Kaggle)
│   └── processed/               Cached master feature table
├── docs/
│   ├── PROJECT_JOURNAL.md       Decision log with rationale
│   ├── PROJECT_ARCHITECTURE.md  System design
│   ├── DATA_DICTIONARY.md       All 7 tables documented
│   ├── FEATURE_ENGINEERING.md   Every feature: formula + business meaning
│   ├── MODELING_DECISIONS.md    Model selection and evaluation rationale
│   ├── DEPLOYMENT_GUIDE.md      Setup and deployment instructions
│   └── FUTURE_IMPROVEMENTS.md  Roadmap and known limitations
├── src/
│   ├── data/
│   │   ├── loader.py            Load all 7 CSV tables
│   │   └── validator.py         Schema validation
│   ├── features/
│   │   ├── application.py       25+ features from main table
│   │   ├── bureau.py            30+ features from bureau tables
│   │   ├── previous.py          15+ features from previous applications
│   │   ├── installments.py      17+ features from payment records
│   │   ├── pos_cash.py          15+ features from POS balances
│   │   ├── credit_card.py       20+ features from CC balances
│   │   └── pipeline.py          Orchestrates all feature joins
│   ├── models/
│   │   ├── train.py             5-fold CV + multi-model training
│   │   ├── evaluate.py          ROC-AUC, PR-AUC, threshold analysis
│   │   ├── explain.py           SHAP global + per-prediction
│   │   └── tune.py              Optuna hyperparameter search
│   ├── preprocessing.py         sklearn ColumnTransformer pipeline
│   └── predict.py               Inference with v1/v2 artifact support
├── api/
│   ├── main.py                  FastAPI application
│   └── schemas.py               Pydantic v2 request/response schemas
├── app/
│   └── app.py                   Streamlit dashboard (3 tabs)
├── tests/
│   ├── test_features.py         17 feature engineering tests
│   └── test_preprocessing.py   12 preprocessing tests
├── models/
│   ├── v1/                      MVP artifacts (XGBoost, main table only)
│   └── v2/                      Full pipeline artifacts
├── Makefile                     make train / app / api / test
├── Dockerfile                   Multi-stage build (app + api targets)
├── docker-compose.yml           API + Streamlit services
└── requirements.txt
```

---

## Key Engineering Decisions

| Decision | Rationale |
|---|---|
| sklearn Pipeline for preprocessing | Prevents train/serve skew — preprocessor and model serialised as one object |
| Zero-fill for supplementary table NaNs | "No bureau history" ≠ "average bureau history" — 0 is the correct semantic |
| scale_pos_weight=11 over SMOTE | No data modification; equivalent upweighting effect; avoids leakage risk |
| F-beta (β=2) threshold optimisation | Catching defaults is 2× more important than precision — reflects business cost model |
| SHAP over permutation importance | Theoretically grounded; not biased by correlated features |
| OrdinalEncoder over OneHotEncoder | 58-category ORGANIZATION_TYPE → 1 column vs. 57; trees handle ordinals natively |

---

## Documentation

All decisions, trade-offs, and alternatives are documented in `docs/`:

- [Project Journal](docs/PROJECT_JOURNAL.md) — what changed and why
- [Architecture](docs/PROJECT_ARCHITECTURE.md) — system design
- [Data Dictionary](docs/DATA_DICTIONARY.md) — all 7 tables
- [Feature Engineering](docs/FEATURE_ENGINEERING.md) — every feature explained
- [Modeling Decisions](docs/MODELING_DECISIONS.md) — model selection rationale
- [Deployment Guide](docs/DEPLOYMENT_GUIDE.md) — setup instructions
- [Future Improvements](docs/FUTURE_IMPROVEMENTS.md) — roadmap

---

## Tech Stack

`Python 3.11` · `pandas` · `scikit-learn` · `XGBoost` · `LightGBM` · `CatBoost` · `SHAP` · `Optuna` · `FastAPI` · `Pydantic v2` · `Streamlit` · `Docker` · `pytest` · `GitHub Actions`
