# Future Improvements

**Project:** Home Loan Default Prediction  
**Last Updated:** 2026-06-04

---

## Priority 1 — High Impact (Next 2 Sprints)

### 1.1 Model Stacking / Ensemble

**What:** Train XGBoost and LightGBM independently, stack their out-of-fold predictions as inputs to a logistic regression meta-learner.

**Why:** Stacking orthogonal models (XGBoost leaf-wise vs. LightGBM depth-wise) typically adds +0.002–0.005 AUC with minimal additional complexity.

**Risk:** Adds a second layer of models to explain, serialize, and monitor. Only worth it if base model AUC is already strong (>0.79).

---

### 1.2 Feature Store Integration

**What:** Instead of recomputing all aggregations from raw CSVs on every training run, cache the processed feature table to data/processed/master_train.csv and implement incremental updates for new rows.

**Why:** Training currently requires loading 30M+ rows from bureau_balance and POS_CASH just to aggregate them. With a feature store, only new applicant rows need processing.

**Technology options:** Feast (open source), Tecton (managed), or a simple DVC-tracked parquet file.

---

### 1.3 Model Monitoring

**What:** Track the distribution of model inputs (feature drift) and outputs (prediction drift) over time. Alert when distributions shift significantly.

**Why:** A model trained on 2016–2018 Home Credit data may degrade as economic conditions change. Without monitoring, this degradation is invisible until business KPIs deteriorate.

**Implementation:** Evidently AI (open source), or manual PSI (Population Stability Index) computation on key features.

---

### 1.4 Calibration

**What:** Apply Platt scaling or isotonic regression to calibrate model probabilities so that "30% default probability" actually means 30% of those predictions are defaults.

**Why:** Raw XGBoost probabilities are often overconfident. Calibrated probabilities are required if they will be used for business decision-making (e.g., pricing, risk bands).

**Implementation:** `sklearn.calibration.CalibratedClassifierCV`

---

## Priority 2 — Medium Impact (Next Quarter)

### 2.1 Automated Retraining Pipeline

**What:** Scheduled retraining (weekly/monthly) with automated evaluation. If new model beats current model in AUC, promote it; otherwise keep current.

**Technology:** GitHub Actions cron + MLflow for experiment tracking and model registry.

---

### 2.2 Risk Banding Instead of Binary Output

**What:** Instead of Low/High risk, segment into 5 risk bands (Very Low / Low / Medium / High / Very High) with associated approval rates and expected default rates.

**Why:** Loan officers need gradations. A "High Risk" applicant at 25% default probability needs different handling than one at 45%.

**Implementation:** Find 4 probability thresholds that create 5 segments with equal population or equal precision levels.

---

### 2.3 A/B Testing Framework

**What:** Deploy two model versions simultaneously and route traffic between them, comparing business outcomes rather than just AUC.

**Why:** AUC is a proxy metric. The true metric is: do approved loans actually repay? A/B testing closes the feedback loop.

---

### 2.4 Fairness Audit

**What:** Check model performance across demographic groups (gender, age range, region) using metrics like equalized odds and demographic parity.

**Why:** Credit models in many jurisdictions must demonstrate non-discrimination. Hiring managers in regulated industries will ask about this.

**Tools:** fairlearn, Aequitas

---

## Priority 3 — Low Impact / Nice to Have

### 3.1 Interactive EDA Dashboard

**What:** Separate Streamlit page with interactive charts: feature distributions, default rates by segment, correlation matrix.

**Why:** Demonstrates data understanding to portfolio reviewers who may not run code.

---

### 3.2 Batch Prediction API

**What:** POST /predict/batch endpoint accepting a CSV upload, returning predictions for all rows.

**Why:** Real lending systems make predictions on batches of applications, not one at a time.

---

### 3.3 Model Comparison Dashboard

**What:** Tab in the Streamlit app showing ROC curves, PR curves, and feature importance charts for all trained models side-by-side.

**Why:** Demonstrates the model selection process, not just the final result.

---

### 3.4 MLflow Integration

**What:** Log all experiments (hyperparameters, metrics, artifacts) to MLflow tracking server.

**Why:** Standard MLOps practice; shows experiment tracking discipline; makes the model comparison process reproducible.

---

## Known Limitations

1. **Data recency:** The Home Credit dataset is from 2018. Model may not generalize to current economic conditions.

2. **Geographic specificity:** Data is from a specific country. Credit behavior patterns may differ significantly in other markets.

3. **No real-time data:** The model uses only application-time features. A production system would benefit from real-time bureau lookups.

4. **Single institution bias:** Bureau features reflect behavior at other institutions as reported to the bureau. Self-reported income and employment are not verified.

5. **Survivorship bias in previous_application:** Only Home Credit customers with previous applications appear in this table. First-time borrowers are underrepresented in the historical behavior features.

6. **Frozen threshold:** The optimal threshold is computed once at training time and stored. As the model's score distribution shifts with new data, the threshold should be reoptimized.
