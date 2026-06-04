# Project Journal — Home Loan Default Prediction

**Project:** Home Loan Default Prediction System  
**Dataset:** Home Credit Default Risk (Kaggle)  
**Engineer:** Shadaf  
**Started:** 2026-06-04  
**Status:** Active — Phase 1–4 Complete

---

## Purpose of This Document

This journal is the institutional memory of the project. Every major decision, architectural change, finding, and milestone is recorded here with its rationale. Future reviewers — including hiring managers and collaborators — should be able to understand not just what was built, but why every choice was made.

---

## 2026-06-04 — Phase 1: Project Audit

### Audit Summary

The MVP was built with a clean, minimal architecture. A gap analysis between claimed capabilities and actual code state revealed the following:

| Claimed Capability | Actual Code State | Risk |
|---|---|---|
| Logistic Regression + RF trained | Only XGBoost in src/train.py | Misleading — no comparison exists |
| Feature engineering (3 ratios) | Not present in preprocessing.py or train.py | Ratios existed only in app.py UI logic |
| EDA completed | Notebook exists, not wired to pipeline | Findings not codified |
| Modular architecture | 4 functions in one preprocessing.py file | Flat, not modular |
| SHAP explainability | In requirements.txt, never used | Dead dependency |

### Technical Debt Identified

1. **Single table usage** — Only application_train.csv used; 6 supplementary tables with rich signal completely ignored
2. **No sklearn.Pipeline** — Preprocessing and model are decoupled; predict.py manually reconstructs inputs, risking silent train/serve divergence
3. **No cross-validation** — Single 80/20 split has high variance; different random seeds shift AUC by ±0.02
4. **No logging** — No visibility into runtime behavior or data quality issues
5. **Hardcoded hyperparameters** — All model parameters embedded in train.py source code
6. **No model versioning** — Retraining silently overwrites pkl artifacts
7. **No input validation** — App accepts any numeric input without range checks
8. **No tests** — Zero regression protection against future changes
9. **No configuration management** — Thresholds, paths, hyperparameters scattered across files

### Strengths Identified

1. Correct XGBoost configuration (scale_pos_weight, early stopping, hist method)
2. Feature defaults at inference time (training medians) — prevents unrealistic predictions on partial input
3. CWD-independent path resolution via get_path()
4. Functional Streamlit app

---

## 2026-06-04 — Phase 1: Architectural Decisions

### Decision 1: Sub-package structure under src/

**Change:** Introduce src/data/, src/features/, src/models/, src/utils/ sub-packages  
**Reason:** src/ grows to 12+ files across data loading, feature engineering, modeling, and utilities. A flat structure becomes unnavigable at scale.  
**Alternative considered:** Keep flat src/ with longer filenames (e.g., src/feature_bureau.py)  
**Why rejected:** Flat structure doesn't scale past 8 files; sub-packages are the standard in production ML repositories and signal engineering maturity to reviewers  
**Backward compatibility:** src/preprocessing.py, src/train.py, src/predict.py retained and refactored in place

### Decision 2: YAML configuration management

**Change:** Extract all hyperparameters, thresholds, and paths to configs/model_config.yaml  
**Reason:** Hyperparameters in source code must be changed in the source and re-committed. YAML config allows experimentation without touching source files, makes all tunables visible in one place, and is standard in MLOps workflows.  
**File:** configs/model_config.yaml

### Decision 3: Structured logging via Python logging module

**Change:** Replace all print() calls with structured logging (src/utils/logger.py)  
**Reason:** print() is invisible to log aggregators, has no severity levels, and cannot be toggled. Structured logs are essential for debugging production systems.  
**Format:** Timestamp + module + level + message

### Decision 4: 5-fold Stratified Cross-Validation as standard evaluation

**Change:** Replace single train_test_split with StratifiedKFold(n_splits=5)  
**Reason:** A single random split has variance. With 307K rows and 8% default rate, different seeds shift AUC by ±0.015–0.025. CV gives mean ± std — the statistically defensible metric for model comparison.  
**Trade-off:** ~5× training time during model selection phase. Acceptable given dataset size and importance of reliable estimates.

### Decision 5: Join all 6 supplementary tables

**Change:** Build aggregation modules for bureau, bureau_balance, previous_application, installments_payments, POS_CASH_balance, credit_card_balance  
**Reason:** These tables contain the applicant's behavioral credit history — the most predictive signal for default risk. Using only the application table is equivalent to underwriting without a credit report.  
**Expected AUC gain:** +0.05 to +0.08 based on Kaggle competition solutions (from ~0.74 to ~0.80+)  
**Risk:** Adds complexity and join-time data leakage risk if not done carefully (all joins are left joins on SK_ID_CURR, an applicant-level key)

---

## Phase Completion Log

| Phase | Date | Status | Key Output |
|---|---|---|---|
| 1 — Project Audit | 2026-06-04 | Complete | This journal + architecture docs |
| 2 — Data Understanding | 2026-06-04 | Complete | src/data/loader.py, src/data/validator.py |
| 3 — Data Integration | 2026-06-04 | Complete | 6 feature modules + pipeline.py |
| 4 — Feature Engineering | 2026-06-04 | Complete | src/features/application.py, 120+ features |
| 5 — Preprocessing Optimization | 2026-06-04 | Complete | Refactored preprocessing.py |
| 6 — Modeling | Pending | — | src/models/train.py + compare all models |
| 7 — Explainability | Pending | — | src/models/explain.py + SHAP in app |
| 8 — Deployment | Pending | — | FastAPI + Docker |
| 9 — Portfolio Optimization | Pending | — | README + CI + demo |
| 10 — Final Review | Pending | — | End-to-end test + final docs |

---

## Open Questions

- What is the business cost of a false negative (missed default) vs. false positive (wrongly rejected applicant)?
  - Drives threshold optimization: higher recall = lower threshold, more false positives
  - Standard in banking: false negatives (bad loans approved) cost 5–10× more than false positives
- Should the model produce a risk band (Low/Medium/High) rather than a binary decision?
  - Useful for loan officer review workflows
  - Planned for Phase 9 dashboard improvements
