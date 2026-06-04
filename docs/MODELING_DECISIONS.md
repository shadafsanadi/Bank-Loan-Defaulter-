# Modeling Decisions

**Project:** Home Loan Default Prediction  
**Last Updated:** 2026-06-04

---

## Problem Framing

**Task type:** Binary classification (default = 1, repay = 0)  
**Primary metric:** ROC-AUC — chosen because it measures ranking ability across all thresholds, which is appropriate when the optimal operating threshold is determined separately by business policy  
**Secondary metric:** PR-AUC (Average Precision) — more informative than ROC-AUC for imbalanced datasets; reflects performance specifically on the minority (default) class  
**Class balance:** ~91.9% repay, ~8.1% default — moderate imbalance

---

## Class Imbalance Strategy

### Why scale_pos_weight, not SMOTE

**Decision:** Use scale_pos_weight = (n_negative / n_positive) ≈ 11 for tree-based models; class_weight="balanced" for Logistic Regression.

**Why not SMOTE:**
- SMOTE generates synthetic minority-class samples via interpolation
- If applied before splitting, synthetic samples from training rows appear in the validation set → data leakage inflating CV scores
- If applied correctly (inside each fold's training set only), adds pipeline complexity with marginal AUC gain for gradient boosting models
- scale_pos_weight achieves the same effect (making the model penalize false negatives more heavily) without data modification

**Why not class_weight for XGBoost:**
- XGBoost does not natively support class_weight; scale_pos_weight is the correct parameter
- scale_pos_weight = 11 means each positive (default) sample is weighted 11× during tree construction, effectively upsampling the minority class in the loss function

---

## Model Selection

### Models Evaluated

| Model | Strengths | Weaknesses | Expected AUC Range |
|---|---|---|---|
| Logistic Regression | Interpretable, fast, calibrated | Assumes linear decision boundary, poor with interactions | 0.68–0.72 |
| Random Forest | Handles non-linearity, robust to outliers | Slower than boosting, memory-heavy | 0.72–0.76 |
| XGBoost | State-of-art tabular performance, handles missing values | Slower training than LightGBM | 0.78–0.82 |
| LightGBM | Fastest training, leaf-wise splits, often matches XGBoost | Less interpretable leaf-wise growth | 0.78–0.82 |
| CatBoost | Best for high-cardinality categoricals, no manual encoding | Slowest to train | 0.77–0.81 |

### Why XGBoost as Primary

1. Proven on this specific dataset by Kaggle competition results
2. Native support for missing values (no imputation needed in the model itself)
3. scale_pos_weight for imbalance
4. Fast histogram-based splits (tree_method="hist")
5. Excellent SHAP integration via TreeExplainer
6. Early stopping prevents overfitting without manual epoch selection

### Why LightGBM as Secondary

1. Trains 3–5× faster than XGBoost on large datasets
2. Leaf-wise growth captures complex interactions better in some cases
3. Healthy competition — if LightGBM beats XGBoost in CV, it becomes primary

### Logistic Regression as Baseline

A baseline model that always predicts the majority class (predict repay for everyone) achieves 91.9% accuracy but 0% recall on defaults — useless. Logistic Regression is the true "minimum viable model" baseline. If XGBoost does not significantly outperform LR on ROC-AUC, something is wrong with the feature engineering or data pipeline.

---

## Hyperparameters

All hyperparameters are stored in configs/model_config.yaml and documented here.

### XGBoost

| Parameter | Value | Rationale |
|---|---|---|
| n_estimators | 1000 | Upper bound — early stopping determines actual count |
| max_depth | 6 | Controls tree complexity. 6 allows 4th-order interactions without extreme overfitting |
| learning_rate | 0.05 | Conservative step size — more trees, smaller steps → better generalization |
| subsample | 0.8 | 80% row sampling per tree — reduces variance |
| colsample_bytree | 0.8 | 80% feature sampling per tree — reduces correlation between trees |
| min_child_weight | 5 | Minimum sum of instance weight in a leaf — prevents leaves with very few default samples |
| scale_pos_weight | 11 | Computed as n_negative/n_positive ≈ 91.9/8.1 |
| early_stopping_rounds | 50 | Stop if validation AUC doesn't improve for 50 rounds |
| tree_method | hist | Histogram-based splits — ~3× faster than exact on large datasets |
| eval_metric | auc | Monitor ROC-AUC on validation set during training |

### LightGBM

| Parameter | Value | Rationale |
|---|---|---|
| n_estimators | 1000 | Upper bound with early stopping |
| learning_rate | 0.05 | Same philosophy as XGBoost |
| num_leaves | 63 | 2^(max_depth-1) — equivalent to depth-6 XGBoost tree |
| min_child_samples | 50 | Minimum samples per leaf — prevents very small default-class leaves |
| subsample | 0.8 | Row sampling |
| colsample_bytree | 0.8 | Feature sampling |
| scale_pos_weight | 11 | Same as XGBoost |

---

## Evaluation Strategy

### Train / Validation / Test Split

```
Full dataset (307,511 rows)
    │
    ├── Test set (15%, locked) ──────── Touched ONCE at final evaluation
    │
    └── Development set (85%)
            │
            └── 5-Fold Stratified CV
                    Fold 1–5: train 68%, validate 17%
                    → Report: mean AUC ± std AUC
```

**Why stratified:** With 8.1% default rate, unstratified splits can produce folds with 5–11% defaults, causing high metric variance.

**Why 15% test instead of 20%:** Larger development set gives more reliable CV estimates. 46K test rows is still more than sufficient for statistical significance.

**Cardinal Rule:** The test set is evaluated exactly once, after the final model is fully selected and hyperparameters are frozen. Any evaluation during development uses CV results only.

### Metrics Reported

| Metric | What It Measures | When It Matters |
|---|---|---|
| ROC-AUC | Ranking ability across all thresholds | Model comparison |
| PR-AUC | Performance on minority class specifically | Imbalanced datasets |
| Precision @ 80% Recall | How many good loans rejected if 80% of bad loans are caught | Business decision |
| F1-Score | Harmonic mean of precision and recall at chosen threshold | Operational threshold |
| Confusion Matrix | Absolute counts of TP, FP, TN, FN | Stakeholder communication |

### Threshold Selection

The model outputs a probability (0–1). Converting to a binary decision requires a threshold. 0.5 is almost never optimal for imbalanced problems.

**Approach:**
1. Compute precision-recall curve on validation set
2. Find threshold that maximizes F2-score (recall-weighted: catching defaults is 2× more important than precision, based on the business cost assumption that a missed default costs 5× more than a wrongly rejected applicant)
3. Document chosen threshold in models/v2/metadata.json

---

## Feature Importance

Feature importance is computed in three ways:

1. **XGBoost gain importance** — Measures how much each feature improves the objective function when used for splitting
2. **XGBoost cover importance** — Measures how many samples are affected by each feature's splits
3. **SHAP mean |SHAP value|** — Theoretically grounded; not susceptible to bias from correlated features

SHAP is the primary importance metric because gain and cover can assign high importance to correlated features arbitrarily.

Expected top features (based on Kaggle competition analysis):
- EXT_SOURCE_2, EXT_SOURCE_3 (external credit scores)
- BUREAU_BB_DPD_RATIO_MEAN (historical delinquency rate)
- INSTAL_DPD_MAX (worst payment lateness)
- BUREAU_OVERDUE_SUM (total overdue amount)
- CREDIT_INCOME_RATIO (debt burden)
- AGE_YEARS (demographic risk)
- DAYS_EMPLOYED (employment stability)

---

## Alternative Architectures Considered

### Neural Network (TabNet or simple MLP)

**Why not used:**
- Tabular neural networks consistently underperform gradient boosting on datasets of this size (<1M rows)
- No interpretability advantage (attention weights in TabNet are not equivalent to SHAP values)
- Much longer training time with no AUC benefit
- Would use if dataset were >10M rows or had dense embeddings (text, images)

### Stacking / Ensemble

**Why deferred:**
- Adds complexity and a second layer of models to explain to hiring managers
- Diminishing returns: stacking XGBoost+LightGBM typically adds +0.001–0.003 AUC
- Best addressed in FUTURE_IMPROVEMENTS.md if base model results are strong

### AutoML (H2O, TPOT)

**Why not used:**
- Black-box process doesn't demonstrate ML engineering skill
- Portfolio goal is to show decision-making ability, not to maximize AUC by any means
- Would eliminate the learning and documentation value of manual model selection
