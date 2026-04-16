
# EDA Summary — Home Credit Default Risk
## Project 1: Credit Scoring Engine

---

## Dataset Overview
- Total applicants: 307,511
- Total features: 122 columns (106 numerical, 16 categorical)
- Target variable: TARGET (1 = Default, 0 = Repaid)
- Class distribution: 8.07% default rate — significant class imbalance confirmed

---

## Key Finding 1 — Class Imbalance
The dataset has an 8.07% default rate (24,825 defaulters vs 282,686 repayers).
Accuracy is not a valid metric for this problem.
Evaluation will use Gini coefficient, KS statistic, and AUC-PR.
Class imbalance will be handled using scale_pos_weight in LightGBM.

---

## Key Finding 2 — Missing Value Strategy
67 of 122 columns have missing values.
Three-tier handling strategy defined:
- Low (<10%): 10 columns — impute with median/mode
- Medium (10-50%): 16 columns — impute + IS_MISSING flag
- High (>50%): 41 columns — IS_MISSING flag + individual review

Behavioural insight confirmed: defaulters have on average 3.3 more 
missing fields than repayers. TOTAL_MISSING_COUNT retained as a feature.

---

## Key Finding 3 — Strongest Numerical Predictors
Top features by correlation with TARGET:
1. EXT_SOURCE_3 (0.1789) — strongest individual predictor
2. EXT_SOURCE_2 (0.1605)
3. EXT_SOURCE_1 (0.1553)
4. DAYS_BIRTH (0.0782) — age is a clean monotonic predictor
5. REGION_RATING_CLIENT_W_CITY (0.0609)

AMT_INCOME_TOTAL shows weak standalone correlation — 
debt-to-income ratio will be engineered in Phase 3.

---

## Key Finding 4 — DAYS_EMPLOYED Anomaly
365,243 records in DAYS_EMPLOYED represent pensioners/unemployed.
Count: 55,374 applicants (18% of dataset).
Anomalous group default rate: 5.4% vs 8.66% for normal group.
Action: Create DAYS_EMPLOYED_ANOMALY flag, replace 365243 with NaN,
impute separately with median.

---

## Key Finding 5 — Age as a Monotonic Predictor
Default rate declines cleanly with age across every band:
- 20-25: 11.74% (highest risk)
- 26-30: 11.24%
- 31-35: 9.98%
- 36-40: 8.67%
- 41-50: 7.55%
- 51-60: 5.93%
- 61-70: 4.92% (lowest risk)
Action: Engineer IS_YOUNG_BORROWER flag for applicants under 30.

---

## Key Finding 6 — Income is Non-Linear
Raw income shows weak, non-linear relationship with default.
Middle income bands (20-60th percentile) default more than bottom 20%.
Top 20% income earners default least (6.52%).
Action: Engineer AMT_INCOME_TOTAL_LOG and debt-to-income ratio.

---

## Key Finding 7 — Strongest Categorical Predictors
OCCUPATION_TYPE: 17.15% (Low-skill Laborers) vs 4.83% (Accountants)
NAME_EDUCATION_TYPE: Clean gradient from 10.93% to 1.83%
NAME_HOUSING_TYPE: Rented apartment highest (12.31%), 
                   Office apartment lowest (6.57%)
NAME_INCOME_TYPE: Pensioner (5.39%) vs Working (9.59%)

---

## Key Finding 8 — Protected Attributes
CODE_GENDER: Male 10.14% vs Female 7.00% default rate.
3.14 percentage point gap identified.
Gender will NOT be used as a model feature (ECOA compliance).
Will be used for disparate impact testing in Phase 5.

---

## Encoding Strategy for Phase 3
- Binary encode: NAME_CONTRACT_TYPE, FLAG_OWN_CAR, FLAG_OWN_REALTY
- One-hot encode: CODE_GENDER (fairness only), NAME_INCOME_TYPE,
  NAME_EDUCATION_TYPE, NAME_FAMILY_STATUS, NAME_HOUSING_TYPE
- Target encode: OCCUPATION_TYPE, ORGANIZATION_TYPE
- Drop candidate: WEEKDAY_APPR_PROCESS_START (weak signal, 0.59% range)

---

## Features to Engineer in Phase 3
1. DEBT_TO_INCOME — AMT_CREDIT / AMT_INCOME_TOTAL
2. ANNUITY_TO_INCOME — AMT_ANNUITY / AMT_INCOME_TOTAL
3. CREDIT_TO_GOODS — AMT_CREDIT / AMT_GOODS_PRICE
4. AGE_YEARS — convert DAYS_BIRTH to positive years
5. EMPLOYMENT_YEARS — convert DAYS_EMPLOYED to positive years
6. IS_YOUNG_BORROWER — flag for applicants under 30
7. DAYS_EMPLOYED_ANOMALY — flag for 365243 encoded records
8. DOCS_SUBMITTED_COUNT — sum of all FLAG_DOCUMENT columns
9. TOTAL_MISSING_COUNT — already created, retain for model
10. EXT_SOURCE_MEAN — mean of EXT_SOURCE_1, 2, 3 where available
11. AMT_INCOME_TOTAL_LOG — log transformed income for baseline model
12. IS_MISSING flags for all medium and high missingness columns

---

## Phase 3 Feature Priority Tiers
Tier 1 (engineer with care): EXT_SOURCE_1/2/3, DAYS_BIRTH, 
                              AMT_INCOME_TOTAL, AMT_CREDIT, 
                              DAYS_EMPLOYED, OCCUPATION_TYPE
Tier 2 (handle mechanically): All remaining numerical columns,
                               Low cardinality categoricals
Tier 3 (flag for review): WEEKDAY_APPR_PROCESS_START,
                           Very high missingness property columns
