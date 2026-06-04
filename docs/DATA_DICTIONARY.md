# Data Dictionary

**Project:** Home Loan Default Prediction  
**Dataset:** Home Credit Default Risk  
**Last Updated:** 2026-06-04

---

## Dataset Overview

| Table | Rows | Key Column | Joins To | Description |
|---|---|---|---|---|
| application_train.csv | 307,511 | SK_ID_CURR | (fact table) | One row per loan application — demographics, financials, target |
| bureau.csv | 1,716,428 | SK_ID_CURR, SK_ID_BUREAU | application via SK_ID_CURR | Credits from other institutions reported to credit bureau |
| bureau_balance.csv | 27,299,925 | SK_ID_BUREAU | bureau via SK_ID_BUREAU | Monthly status snapshots for each bureau credit |
| previous_application.csv | 1,670,214 | SK_ID_CURR | application via SK_ID_CURR | Past loan applications at Home Credit |
| installments_payments.csv | 13,605,401 | SK_ID_CURR | application via SK_ID_CURR | Payment records for previous Home Credit loans |
| POS_CASH_balance.csv | 10,001,358 | SK_ID_CURR | application via SK_ID_CURR | Monthly POS and cash loan snapshots |
| credit_card_balance.csv | 3,840,312 | SK_ID_CURR | application via SK_ID_CURR | Monthly credit card snapshots |

**Target variable:** TARGET (in application_train.csv)
- 0 = Loan repaid on time
- 1 = Client had payment difficulties (default)
- Distribution: 91.93% repay, 8.07% default

---

## application_train.csv — Key Features

### Identifiers
| Column | Type | Description |
|---|---|---|
| SK_ID_CURR | int | Unique loan application ID — primary key |
| TARGET | int (0/1) | Prediction target |

### Applicant Demographics
| Column | Type | Description | Notes |
|---|---|---|---|
| CODE_GENDER | str | M / F / XNA | XNA rows are data quality artifacts — removed |
| DAYS_BIRTH | int | Days before application date (negative) | Divide by -365 to get age in years |
| CNT_CHILDREN | int | Number of children | |
| CNT_FAM_MEMBERS | float | Family size | |
| NAME_FAMILY_STATUS | str | Civil marriage / Married / Separated / Single / Widow | |
| NAME_EDUCATION_TYPE | str | Academic degree / Higher education / Incomplete higher / Lower secondary / Secondary | |

### Applicant Financials
| Column | Type | Description | Notes |
|---|---|---|---|
| AMT_INCOME_TOTAL | float | Annual income in local currency | Has extreme outliers (>99th pct capped) |
| AMT_CREDIT | float | Credit amount of the loan | |
| AMT_ANNUITY | float | Annual loan annuity (yearly installment) | |
| AMT_GOODS_PRICE | float | Price of goods for consumer loans | Null for cash loans |
| NAME_INCOME_TYPE | str | Commercial associate / Pensioner / State servant / Unemployed / Working | |

### Loan Characteristics
| Column | Type | Description |
|---|---|---|
| NAME_CONTRACT_TYPE | str | Cash loans / Revolving loans |
| NAME_HOUSING_TYPE | str | Co-op apartment / House or apartment / Municipal apartment / Office apartment / Rented apartment / With parents |
| REGION_POPULATION_RELATIVE | float | Normalized population of client's region |
| REGION_RATING_CLIENT | int | Home Credit's rating of client region (1/2/3) |
| REGION_RATING_CLIENT_W_CITY | int | Region rating including city |

### Employment and Identity
| Column | Type | Description | Notes |
|---|---|---|---|
| DAYS_EMPLOYED | int | Days before application client started current employment (negative) | 365243 = not employed (sentinel value) |
| DAYS_REGISTRATION | int | Days before application client registered change of registration | |
| DAYS_ID_PUBLISH | int | Days before application client changed ID document | |
| OCCUPATION_TYPE | str | Job type — 18 categories | ~31% missing |
| ORGANIZATION_TYPE | str | Employer organization type — 58 categories | |

### External Credit Scores
| Column | Type | Description | Notes |
|---|---|---|---|
| EXT_SOURCE_1 | float | Normalized external credit score 1 | ~56% missing — from one bureau |
| EXT_SOURCE_2 | float | Normalized external credit score 2 | ~0.3% missing — most complete |
| EXT_SOURCE_3 | float | Normalized external credit score 3 | ~19% missing |

These are among the top 3 most predictive features in the dataset.

### Contact and Social Flags
| Column | Type | Description |
|---|---|---|
| FLAG_MOBIL | int (0/1) | Did client provide mobile phone? |
| FLAG_EMP_PHONE | int (0/1) | Did client provide work phone? |
| FLAG_WORK_PHONE | int (0/1) | Did client provide work phone number? |
| FLAG_CONT_MOBILE | int (0/1) | Was mobile phone reachable? |
| FLAG_PHONE | int (0/1) | Did client provide home phone? |
| FLAG_EMAIL | int (0/1) | Did client provide email? |
| FLAG_OWN_CAR | str (Y/N) | Does client own a car? |
| FLAG_OWN_REALTY | str (Y/N) | Does client own real estate? |

### Document Flags (FLAG_DOCUMENT_2 through FLAG_DOCUMENT_21)
Binary flags indicating whether each document type was provided. Majority have very low positive rates (<10%).

---

## bureau.csv — Key Features

| Column | Type | Description |
|---|---|---|
| SK_ID_CURR | int | Link to application_train |
| SK_ID_BUREAU | int | Unique credit ID at bureau — links to bureau_balance |
| CREDIT_ACTIVE | str | Active / Closed / Bad debt / Sold |
| CREDIT_CURRENCY | str | Currency of bureau credit |
| DAYS_CREDIT | int | Days before application client applied for bureau credit (negative) |
| CREDIT_DAY_OVERDUE | int | Days past due at time of application |
| DAYS_CREDIT_ENDDATE | float | Days before/after application when bureau credit ends |
| AMT_CREDIT_MAX_OVERDUE | float | Maximum overdue amount across the credit life |
| CNT_CREDIT_PROLONG | int | How many times credit was prolonged |
| AMT_CREDIT_SUM | float | Current credit amount |
| AMT_CREDIT_SUM_DEBT | float | Current debt on credit |
| AMT_CREDIT_SUM_LIMIT | float | Current credit limit |
| AMT_CREDIT_SUM_OVERDUE | float | Current amount overdue |
| CREDIT_TYPE | str | Consumer credit / Credit card / Mortgage / Car loan / etc. |
| DAYS_CREDIT_UPDATE | int | Days before application last information received for credit |
| AMT_ANNUITY | float | Annuity of bureau credit |

---

## bureau_balance.csv — Key Features

| Column | Type | Description |
|---|---|---|
| SK_ID_BUREAU | int | Link to bureau.csv |
| MONTHS_BALANCE | int | Month of balance relative to application date (0 = most recent) |
| STATUS | str | 0=No DPD, 1=1-30 DPD, 2=31-60 DPD, 3=61-90 DPD, 4=91-120 DPD, 5=120+DPD, C=Closed, X=Unknown |

---

## previous_application.csv — Key Features

| Column | Type | Description |
|---|---|---|
| SK_ID_CURR | int | Link to application_train |
| SK_ID_PREV | int | Unique previous application ID |
| NAME_CONTRACT_TYPE | str | Cash loans / Consumer loans / Revolving loans / XNA |
| AMT_ANNUITY | float | Annuity of previous application |
| AMT_APPLICATION | float | Amount for which previous application was applied |
| AMT_CREDIT | float | Final credit amount of previous application |
| AMT_DOWN_PAYMENT | float | Down payment for previous application |
| AMT_GOODS_PRICE | float | Goods price in previous application |
| NAME_CONTRACT_STATUS | str | Approved / Canceled / Refused / Unused offer |
| DAYS_DECISION | int | Days before application previous decision was made |
| NAME_PAYMENT_TYPE | str | Payment type for previous loan |
| CODE_REJECT_REASON | str | Reason for rejection (if refused) |
| CNT_PAYMENT | float | Number of installments in previous application |
| RATE_DOWN_PAYMENT | float | Down payment rate |

---

## installments_payments.csv — Key Features

| Column | Type | Description |
|---|---|---|
| SK_ID_CURR | int | Link to application_train |
| SK_ID_PREV | int | Link to previous_application |
| NUM_INSTALMENT_VERSION | float | Version of installment (restructured loans get new version) |
| NUM_INSTALMENT_NUMBER | float | Installment number within the loan |
| DAYS_INSTALMENT | float | When installment was due (days before application) |
| DAYS_ENTRY_PAYMENT | float | When payment was actually made (days before application) |
| AMT_INSTALMENT | float | Installment amount due |
| AMT_PAYMENT | float | Amount actually paid |

**Derived signals:**
- DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT > 0 → late payment
- AMT_PAYMENT < AMT_INSTALMENT → underpayment

---

## POS_CASH_balance.csv — Key Features

| Column | Type | Description |
|---|---|---|
| SK_ID_CURR | int | Link to application_train |
| SK_ID_PREV | int | Link to previous_application |
| MONTHS_BALANCE | int | Month of balance (0 = most recent) |
| CNT_INSTALMENT | float | Installments remaining in loan |
| CNT_INSTALMENT_FUTURE | float | Installments remaining in loan at reporting month |
| NAME_CONTRACT_STATUS | str | Active / Completed / Signed / etc. |
| SK_DPD | int | Days past due during the month |
| SK_DPD_DEF | int | Days past due (default definition) |

---

## credit_card_balance.csv — Key Features

| Column | Type | Description |
|---|---|---|
| SK_ID_CURR | int | Link to application_train |
| SK_ID_PREV | int | Link to previous_application |
| MONTHS_BALANCE | int | Month of balance (0 = most recent) |
| AMT_BALANCE | float | Balance outstanding on credit card |
| AMT_CREDIT_LIMIT_ACTUAL | float | Credit limit during the month |
| AMT_DRAWINGS_ATM_CURRENT | float | Amount drawn from ATM during the month |
| AMT_DRAWINGS_CURRENT | float | Amount drawn during the month |
| AMT_DRAWINGS_POS_CURRENT | float | Amount drawn at POS during the month |
| AMT_INST_MIN_REGULARITY | float | Minimum installment required (minimum payment due) |
| AMT_PAYMENT_CURRENT | float | Amount paid by client during the month |
| AMT_PAYMENT_TOTAL_CURRENT | float | Total amount paid |
| AMT_RECEIVABLE_PRINCIPAL | float | Amount receivable for principal |
| CNT_DRAWINGS_ATM_CURRENT | int | Number of ATM draws |
| CNT_DRAWINGS_CURRENT | int | Number of total draws |
| CNT_INSTALMENT_MATURE_CUM | float | Cumulative count of mature installments |
| NAME_CONTRACT_STATUS | str | Active / Completed / Demand / Signed |
| SK_DPD | int | Days past due during the month |
| SK_DPD_DEF | int | Days past due (default definition) |

---

## Missing Value Summary (application_train.csv — Top Issues)

| Column | Missing % | Handling Strategy |
|---|---|---|
| EXT_SOURCE_1 | 56.4% | Impute with median; missing indicator flag |
| OCCUPATION_TYPE | 31.3% | Impute with mode; "Unknown" category |
| EXT_SOURCE_3 | 19.8% | Impute with median |
| AMT_GOODS_PRICE | 0.1% | Impute with median |
| AMT_ANNUITY | 0.001% | Impute with median |

Columns with >40% missing values (threshold set in HIGH_MISSING_THRESHOLD) are dropped entirely before imputation.

---

## Join Key Relationships

```
application_train.csv (SK_ID_CURR)
    ├──→ bureau.csv (SK_ID_CURR)
    │        └──→ bureau_balance.csv (SK_ID_BUREAU)
    ├──→ previous_application.csv (SK_ID_CURR)
    │        ├──→ installments_payments.csv (SK_ID_PREV)
    │        ├──→ POS_CASH_balance.csv (SK_ID_PREV)
    │        └──→ credit_card_balance.csv (SK_ID_PREV)
    ├──→ installments_payments.csv (SK_ID_CURR)
    ├──→ POS_CASH_balance.csv (SK_ID_CURR)
    └──→ credit_card_balance.csv (SK_ID_CURR)
```

All joins to application_train are LEFT joins: applicants with no bureau history / no previous applications get NaN values for those feature groups, which are then imputed with 0 (indicating "no history of this type").
