# Project Architecture

**Home Loan Default Prediction System**  
**Version:** 2.0 (Post-Audit Refactor)  
**Last Updated:** 2026-06-04

---

## System Overview

This is an end-to-end machine learning system for predicting home loan default risk, built on the Home Credit Default Risk dataset. The architecture follows a layered design separating data ingestion, feature engineering, modeling, serving, and presentation.

---

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                               │
│                                                                  │
│  data/raw/                                                       │
│  ├── application_train.csv   (307,511 rows — main fact table)   │
│  ├── bureau.csv              (1.7M rows — external credit history│
│  ├── bureau_balance.csv      (27M rows — monthly bureau status)  │
│  ├── previous_application.csv(1.7M rows — past HC applications)  │
│  ├── installments_payments.csv(13.6M rows — payment history)     │
│  ├── POS_CASH_balance.csv    (10M rows — POS loan snapshots)     │
│  └── credit_card_balance.csv (3.8M rows — CC monthly snapshots)  │
│                                                                  │
│  src/data/loader.py   → DataLoader class                        │
│  src/data/validator.py → schema + dtype validation              │
└──────────────────────────┬───────────────────────────────────────┘
                           │ raw DataFrames
┌──────────────────────────▼───────────────────────────────────────┐
│                    FEATURE ENGINEERING LAYER                     │
│                                                                  │
│  src/features/                                                   │
│  ├── application.py   → 25+ features from main table            │
│  ├── bureau.py        → 20+ features from bureau tables         │
│  ├── previous.py      → 15+ features from previous_application  │
│  ├── installments.py  → 12+ features from installments          │
│  ├── pos_cash.py      → 10+ features from POS_CASH_balance      │
│  ├── credit_card.py   → 12+ features from credit_card_balance   │
│  └── pipeline.py      → assembles master feature DataFrame      │
│                                                                  │
│  Output: master DataFrame — one row per applicant, ~300 features │
└──────────────────────────┬───────────────────────────────────────┘
                           │ master DataFrame
┌──────────────────────────▼───────────────────────────────────────┐
│                    PREPROCESSING LAYER                           │
│                                                                  │
│  src/preprocessing.py                                            │
│  ├── ColumnTransformer (numeric: median impute + scale)         │
│  ├── ColumnTransformer (categorical: mode impute + ordinal enc) │
│  ├── Column selection and outlier capping                        │
│  └── sklearn.Pipeline wraps preprocessor + model               │
└──────────────────────────┬───────────────────────────────────────┘
                           │ preprocessed arrays
┌──────────────────────────▼───────────────────────────────────────┐
│                       MODELING LAYER                             │
│                                                                  │
│  src/models/                                                     │
│  ├── train.py    → 5-fold CV, multi-model comparison, final fit  │
│  ├── evaluate.py → ROC-AUC, PR-AUC, threshold analysis          │
│  ├── explain.py  → SHAP TreeExplainer, global + local plots     │
│  └── tune.py     → Optuna hyperparameter search                 │
│                                                                  │
│  models/                                                         │
│  ├── v1/ (MVP artifacts)                                        │
│  └── v2/ (post-refactor artifacts)                              │
│      ├── pipeline.pkl     (preprocessor + model)                │
│      ├── feature_names.pkl                                       │
│      └── metadata.json   (AUC, threshold, training date)        │
└──────────────────────────┬───────────────────────────────────────┘
                           │ model artifacts
         ┌─────────────────┴──────────────────┐
         │                                     │
┌────────▼──────────┐              ┌───────────▼──────────────────┐
│   SERVING LAYER   │              │        UI LAYER              │
│                   │              │                              │
│  api/main.py      │              │  app/app.py (Streamlit)      │
│  FastAPI          │              │  ├── Sidebar: customer input  │
│  POST /predict    │◄─────────────│  ├── Prediction: risk score  │
│  GET  /health     │              │  ├── SHAP waterfall plot     │
│  GET  /explain    │              │  ├── Model performance tab   │
│                   │              │  └── Business context tab    │
└───────────────────┘              └──────────────────────────────┘
```

---

## Data Flow

```
1. Raw CSV files → DataLoader.load_all()
       ↓
2. Per-table aggregations → bureau.py, previous.py, installments.py, etc.
       ↓
3. Left join all aggregations onto application_train on SK_ID_CURR
       ↓
4. Application feature engineering → application.py
       ↓
5. Master DataFrame saved to data/processed/master_train.csv
       ↓
6. sklearn.Pipeline(preprocessor + model).fit(X_train, y_train)
       ↓
7. Pipeline saved to models/v2/pipeline.pkl
       ↓
8. At inference: pipeline.predict_proba(input_df) → probability
       ↓
9. SHAP waterfall plot generated from TreeExplainer
       ↓
10. Result returned to API or Streamlit app
```

---

## Technology Choices

| Component | Technology | Rationale |
|---|---|---|
| Data processing | pandas | Standard for tabular ML; excellent for groupby aggregations |
| ML pipeline | scikit-learn Pipeline | Prevents train/serve skew; serializable as one artifact |
| Primary model | XGBoost / LightGBM | Best-in-class for tabular data; handles imbalance via scale_pos_weight |
| Explainability | SHAP TreeExplainer | Theoretically grounded (Shapley values); native XGBoost support |
| Hyperparameter search | Optuna | TPE sampler outperforms grid/random search; pruning for efficiency |
| Configuration | YAML | Human-readable, version-controllable, separates config from code |
| Logging | Python logging | Standard; compatible with all log aggregators |
| Serving | FastAPI | Async, auto-documented, Pydantic validation, production-grade |
| UI | Streamlit | Rapid ML dashboard development; sufficient for portfolio demo |
| Containerization | Docker + docker-compose | Reproducible environment; deployment-ready |
| Testing | pytest | Standard Python testing framework |
| CI | GitHub Actions | Free for public repos; runs tests on every push |

---

## Module Responsibilities

### src/data/loader.py
- Single responsibility: load raw CSV files from data/raw/
- No transformation logic — transformation is in feature modules
- Returns dict of raw DataFrames

### src/data/validator.py
- Validates expected columns, dtypes, and row counts
- Raises informative errors if data is corrupted or partially downloaded
- Not in the hot path; run once before training

### src/features/application.py
- All feature engineering from application_train.csv
- Ratio features, age/employment features, document flags
- External score combinations

### src/features/bureau.py
- Aggregates bureau_balance to per-SK_ID_BUREAU metrics
- Aggregates bureau + bureau_balance to per-SK_ID_CURR metrics
- Captures external credit history: overdue amounts, DPD months, active loan counts

### src/features/pipeline.py
- Orchestrates the full feature build: loads tables, calls each module, joins results
- Entry point for both training and batch inference
- Saves master DataFrame to data/processed/

### src/models/train.py
- Loads master features, runs 5-fold CV for model comparison
- Trains final model on full training data after comparison
- Saves pipeline artifact + metadata

### src/models/evaluate.py
- Computes ROC-AUC, PR-AUC, classification report
- Generates threshold analysis (precision/recall tradeoff curve)
- Generates calibration plot

### src/models/explain.py
- Computes SHAP values using TreeExplainer
- Generates global summary plot (feature importance)
- Generates per-prediction waterfall plot for app

### api/main.py
- FastAPI application with input validation via Pydantic
- POST /predict: returns prediction + probability + SHAP explanation
- GET /health: model version + uptime check

---

## Scalability Considerations

- **Feature modules are stateless** — each takes a raw DataFrame and returns a flat aggregated DataFrame; they can be parallelized with multiprocessing if tables grow very large
- **sklearn.Pipeline** makes the preprocessing-model bundle atomically serializable and loadable in any environment
- **YAML config** allows A/B testing different model configurations without code changes
- **FastAPI async** handles concurrent prediction requests without blocking
- **Docker** eliminates "works on my machine" deployment failures

---

## Known Limitations

- Current design is batch-training, not online learning — model must be retrained to incorporate new data
- No feature store — features are recomputed from raw CSVs on each training run
- No model monitoring (data drift, prediction distribution drift) — planned for future phase
- Single-region deployment assumed — no distributed inference infrastructure
