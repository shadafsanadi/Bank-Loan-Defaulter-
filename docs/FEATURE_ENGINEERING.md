# Feature Engineering Documentation

**Project:** Home Loan Default Prediction  
**Last Updated:** 2026-06-04

---

## Overview

Features are organized by source table. Each feature is documented with its formula, source columns, business meaning, expected predictive value, and any assumptions or risks.

Total engineered features: ~120 (across all 7 tables)  
Raw features from application_train.csv: ~80  
Combined feature space after joining: ~300

---

## Source: application_train.csv

### Credit Burden Features

| Feature | Formula | Business Meaning | Expected Predictive Value |
|---|---|---|---|
| CREDIT_INCOME_RATIO | AMT_CREDIT / AMT_INCOME_TOTAL | What fraction of annual income is the total loan? Higher = more financial stress | **High** — debt-to-income is a primary underwriting metric |
| ANNUITY_INCOME_RATIO | AMT_ANNUITY / AMT_INCOME_TOTAL | What fraction of income goes to monthly installments? | **High** — payment burden relative to income |
| CREDIT_TERM_MONTHS | AMT_CREDIT / AMT_ANNUITY | Loan duration in months — longer terms increase risk exposure | **Medium** — longer loans have more opportunity to default |
| GOODS_CREDIT_RATIO | AMT_GOODS_PRICE / AMT_CREDIT | Ratio of goods value to credit amount. <1 means over-financed (includes fees/insurance) | **Medium** — gap between financed amount and collateral value |

**Assumptions:** AMT_INCOME_TOTAL and AMT_ANNUITY are never exactly zero (enforced with +1 denominator). Extreme outliers (income > 99th percentile) are capped before ratio computation to prevent division stability issues.

---

### Age and Employment Features

| Feature | Formula | Business Meaning | Expected Predictive Value |
|---|---|---|---|
| AGE_YEARS | -DAYS_BIRTH / 365 | Applicant age. Younger applicants historically have higher default rates | **High** — age is one of the strongest demographic risk signals |
| EMPLOYED_YEARS | -DAYS_EMPLOYED / 365 | Years at current employer. Longer tenure = more stability | **High** — employment stability is a core creditworthiness signal |
| EMPLOYMENT_AGE_RATIO | EMPLOYED_YEARS / AGE_YEARS | Fraction of working life at current job. High ratio = stable career | **Medium** — controls for age when interpreting employment length |
| IS_UNEMPLOYED | 1 if DAYS_EMPLOYED == 365243 else 0 | DAYS_EMPLOYED=365243 is a sentinel value in this dataset meaning "pensioner or not working" | **High** — unemployment is a direct default risk factor |

**Note on IS_UNEMPLOYED:** DAYS_EMPLOYED = 365,243 (~1000 years) is a coding convention in this dataset for applicants who are not employed. Failing to handle this sentinel would produce a wildly incorrect EMPLOYED_YEARS value. After flagging, DAYS_EMPLOYED is set to NaN for these rows.

---

### External Credit Score Features

| Feature | Formula | Business Meaning | Expected Predictive Value |
|---|---|---|---|
| EXT_SOURCE_MEAN | mean(EXT_SOURCE_1, 2, 3) | Average normalized credit score from 3 external bureaus | **Very High** — external scores are top-3 most predictive features in this dataset |
| EXT_SOURCE_PRODUCT | EXT_SOURCE_2 × EXT_SOURCE_3 | Interaction: both scores must be high for product to be high | **High** — captures compounding of good credit signals |
| EXT_SOURCE_STD | std(EXT_SOURCE_1, 2, 3) | Disagreement between bureaus — high std may indicate data quality issues | **Medium** — inconsistent scores can signal unreliable reporting |
| EXT_SOURCE_MIN | min(EXT_SOURCE_1, 2, 3) | Worst score across all bureaus — captures the weakest link | **High** — a single bad bureau score can indicate hidden risk |
| CREDIT_INCOME_x_EXT2 | CREDIT_INCOME_RATIO × EXT_SOURCE_2 | Interaction: high debt burden + low creditworthiness = compounded risk | **Medium** — captures the worst-case combination |

---

### Document and Contact Features

| Feature | Formula | Business Meaning | Expected Predictive Value |
|---|---|---|---|
| DOCS_PROVIDED_COUNT | sum(FLAG_DOCUMENT_*) | Number of supporting documents provided with application | **Low-Medium** — more documentation may indicate compliance and transparency |
| CONTACT_CHANNELS_COUNT | sum(FLAG_MOBIL, FLAG_PHONE, etc.) | Number of active contact channels. More channels = easier to reach for collections | **Low** — weak signal but costs nothing to include |

---

## Source: bureau.csv + bureau_balance.csv

These tables capture the applicant's credit history at OTHER financial institutions as reported to the credit bureau. This is analogous to a traditional credit report.

### Bureau Loan Count Features

| Feature | Source | Business Meaning | Expected Predictive Value |
|---|---|---|---|
| BUREAU_LOAN_COUNT | count(SK_ID_BUREAU) | Total number of previous credits at other institutions | **Medium** — high count may indicate credit dependency |
| BUREAU_ACTIVE_COUNT | sum(IS_ACTIVE) | Number of currently open credits | **High** — high active count = high current debt burden |
| BUREAU_CLOSED_COUNT | sum(IS_CLOSED) | Successfully closed credits — track record of repayment | **Medium** — more closures = more repayment history |
| BUREAU_BAD_DEBT_COUNT | sum(IS_BAD_DEBT) | Credits classified as bad debt | **Very High** — direct evidence of past default |
| BUREAU_ACTIVE_RATIO | BUREAU_ACTIVE_COUNT / BUREAU_LOAN_COUNT | Fraction of credits still open — high ratio = heavy current obligations | **High** |

### Bureau Amount Features

| Feature | Source | Business Meaning | Expected Predictive Value |
|---|---|---|---|
| BUREAU_DEBT_SUM | sum(AMT_CREDIT_SUM_DEBT) | Total outstanding debt across all bureau credits | **High** — aggregate debt burden |
| BUREAU_CREDIT_SUM | sum(AMT_CREDIT_SUM) | Total credit limit across all bureau credits | **Medium** — total access to credit |
| BUREAU_OVERDUE_SUM | sum(AMT_CREDIT_SUM_OVERDUE) | Total overdue amount across all bureau credits | **Very High** — current delinquency is the strongest default predictor |
| BUREAU_OVERDUE_MAX | max(AMT_CREDIT_SUM_OVERDUE) | Worst single overdue balance | **High** — magnitude of worst delinquency |
| BUREAU_DEBT_CREDIT_RATIO | BUREAU_DEBT_SUM / BUREAU_CREDIT_SUM | Utilization across all bureau credits | **High** — high utilization signals financial stress |
| BUREAU_MAX_DPD | max(CREDIT_DAY_OVERDUE) | Maximum days past due ever recorded | **Very High** — worst payment lateness in history |

### Bureau Balance (Monthly Status) Features

| Feature | Source | Business Meaning | Expected Predictive Value |
|---|---|---|---|
| BUREAU_BB_DPD_MONTHS_SUM | sum(BB_DPD_MONTHS_COUNT) | Total number of months with any payment delinquency | **Very High** — frequency of past delinquency |
| BUREAU_BB_DPD_RATIO_MEAN | mean(BB_DPD_RATIO) | Fraction of months with delinquency per credit, averaged | **High** — systematic vs. isolated delinquency |
| BUREAU_BB_STATUS_MAX | max(BB_STATUS_MAX) | Worst delinquency severity ever (0=current, 5=120+ DPD) | **Very High** — worst-case payment behavior |
| BUREAU_BB_SEVERE_DPD_SUM | sum(BB_SEVERE_DPD_MONTHS) | Total months with severe delinquency (61+ DPD) | **Very High** — severe delinquency = near-default behavior |

---

## Source: previous_application.csv

Captures the applicant's history of past loan applications AT HOME CREDIT specifically.

| Feature | Business Meaning | Expected Predictive Value |
|---|---|---|
| PREV_APP_COUNT | Total previous applications — frequent applicants may be credit-dependent | Medium |
| PREV_APPROVED_COUNT | Approved applications — track record of creditworthiness | Medium |
| PREV_REFUSED_COUNT | Refused applications — past rejections signal elevated risk | High |
| PREV_APPROVED_RATIO | Approval rate — high refusal rate is a red flag | High |
| PREV_CONSUMER_LOAN_COUNT | Consumer loan history | Low-Medium |
| PREV_CASH_LOAN_COUNT | Cash loan history — cash loans have higher default rates than consumer loans | Medium |
| PREV_MAX_AMT_CREDIT | Largest previous credit — history with large loans | Medium |
| PREV_AMT_CREDIT_MEAN | Average credit amount in past applications | Medium |
| PREV_AMT_DOWN_PAYMENT_MEAN | Average down payment — willingness to contribute equity | Medium |
| PREV_DAYS_DECISION_MEAN | Average days between application and decision | Low |
| PREV_CNT_PAYMENT_MEAN | Average number of installments in past loans | Low-Medium |
| PREV_RATE_DOWN_PAYMENT_MEAN | Average down payment rate — higher = lower LTV = lower risk | Medium |

---

## Source: installments_payments.csv

Actual repayment behavior on all previous Home Credit loans — the most behaviorally rich signal.

| Feature | Business Meaning | Expected Predictive Value |
|---|---|---|
| INSTAL_COUNT | Total number of installment records | Low — just volume |
| INSTAL_DPD_MAX | Maximum days past due on any payment ever | **Very High** — worst payment behavior |
| INSTAL_DPD_MEAN | Average days past due across all payments | **Very High** — systematic lateness pattern |
| INSTAL_DPD_STD | Variability in payment timing | High — erratic payers are higher risk |
| INSTAL_LATE_PAYMENT_COUNT | Number of payments made after due date | High |
| INSTAL_LATE_PAYMENT_RATIO | Fraction of payments made late | Very High |
| INSTAL_AMT_PAYMENT_DEFICIT_MEAN | Average shortfall between amount owed and amount paid | High — underpayment behavior |
| INSTAL_AMT_PAYMENT_DEFICIT_SUM | Total payment deficit | High |
| INSTAL_AMT_PAYMENT_PERC_MEAN | Average payment / installment amount — close to 1.0 = good payer | Very High |
| INSTAL_DAYS_ENTRY_PAYMENT_MEAN | Days before/after due date (negative = early, positive = late) | Very High |
| INSTAL_NUM_INSTALMENT_VERSION_MAX | Number of version changes — renegotiated loans may indicate financial stress | Medium |

---

## Source: POS_CASH_balance.csv

Monthly snapshots of point-of-sale and cash loan balances at Home Credit.

| Feature | Business Meaning | Expected Predictive Value |
|---|---|---|
| POS_COUNT | Total monthly records | Low |
| POS_MONTHS_BALANCE_MEAN | Average remaining months | Low-Medium |
| POS_SK_DPD_MAX | Maximum days past due on POS loans | Very High |
| POS_SK_DPD_MEAN | Average DPD on POS loans | Very High |
| POS_SK_DPD_DEF_MAX | Maximum days past due (default definition) | Very High |
| POS_COMPLETED_COUNT | Number of completed POS contracts | Medium |
| POS_ACTIVE_COUNT | Number of active POS contracts | Medium |
| POS_COMPLETED_RATIO | Fraction of contracts completed — good track record | High |
| POS_LATE_PAYMENTS | Count of months with any DPD | High |

---

## Source: credit_card_balance.csv

Monthly credit card account snapshots — reveals spending and payment behavior.

| Feature | Business Meaning | Expected Predictive Value |
|---|---|---|
| CC_COUNT | Total monthly records | Low |
| CC_UTILIZATION_MEAN | Mean(balance / credit limit) — credit utilization is a core risk signal | **Very High** |
| CC_UTILIZATION_MAX | Maximum utilization — peak financial stress | High |
| CC_DPD_MAX | Maximum days past due on credit card | Very High |
| CC_DPD_MEAN | Average DPD | Very High |
| CC_DRAWINGS_ATM_MEAN | Average ATM cash withdrawals — cash advances signal financial distress | High |
| CC_DRAWINGS_TOTAL_MEAN | Average total drawings | Medium |
| CC_AMT_PAYMENT_TOTAL_CURRENT_MEAN | Average total payments | Medium |
| CC_AMT_BALANCE_MEAN | Average outstanding balance | Medium |
| CC_AMT_CREDIT_LIMIT_ACTUAL_MEAN | Average credit limit | Low-Medium |
| CC_SK_DPD_DEF_MAX | Max DPD by default definition | Very High |
| CC_MIN_PAYMENT_RATIO | Fraction of months where only minimum payment made | High |

---

## Feature Engineering Principles Applied

1. **No leakage:** All features use only information available at loan application time. Temporal features (installments, bureau_balance) are aggregated over the full history — no future information is used.

2. **Sentinel value handling:** DAYS_EMPLOYED = 365,243 is explicitly converted to IS_UNEMPLOYED flag before any numerical operations.

3. **Division safety:** All ratio denominators have +1 added to prevent division by zero while minimally affecting the ratio value.

4. **Group-level aggregation:** All supplementary tables are aggregated to SK_ID_CURR level BEFORE joining to the main table, ensuring one row per applicant with no row duplication.

5. **Missing values from joins:** Applicants with no bureau/previous history get NaN for bureau/previous features. These are imputed with 0 (meaning "no history of this type") rather than median, which would imply some history.
