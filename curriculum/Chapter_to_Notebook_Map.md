---
title: Chapter to Notebook Cell Map
---

# Chapter to Notebook Cell Map

**Purpose:** Open a chapter from the curriculum and the matching notebook side by side. This maps every chapter to the exact cells that implement it, so you can read the theory and immediately check the real code.

**Correction from earlier materials:** the train/validation/test split IS stratified — confirmed directly in `03_model_training.ipynb` cell 4: `stratify=y`, a 60/20/20 split, `random_state=42`. Earlier docs listed this as an open question; it's now settled.

---

## Quick Lookup — Chapter to Cells

| Chapter | Notebook | Cells |
|---|---|---|
| 0 — Problem Framing | *(none — conceptual, before any code)* | — |
| 1 — Data Loading & First Inspection | 01_eda.ipynb | 0–4, 13 |
| 2 — EDA: Distributions, Missingness, Imbalance, Correlation | 01_eda.ipynb | 5–8, 10–12, 14–17 |
| 3 — Feature Engineering & Encoding | 02_feature_engineering.ipynb | 0–60 (whole notebook) |
| 4 — Train/Validation/Test Split & Imputation | 03_model_training.ipynb | 0–7 (cell 7 shared with Ch. 5) |
| 5 — Baseline Model: Logistic Regression | 03_model_training.ipynb | 7–9 (cell 7 shared with Ch. 4) |
| 6 — Main Model: LightGBM | 03_model_training.ipynb | 10, 11, 13, 14 |
| 7 — Evaluation: AUC, Gini, AUC-PR, KS | 03_model_training.ipynb | 8, 12 |
| 8 — Interpretability: SHAP | *(not built — no cells exist yet)* | — |
| 9 — Dashboard: Streamlit | *(not built — app/streamlit_app.py is empty)* | — |

Cell 7 in `03_model_training.ipynb` does double duty: it fits `StandardScaler` (Chapter 4) and instantiates `LogisticRegression(class_weight='balanced')` (Chapter 5) in the same cell — worth noting as one cell, two chapters.

---

## Full Cell-by-Cell Map

### 01_eda.ipynb (19 cells) — Chapters 1–2

| Cell | Title / first line | Chapter |
|---|---|---|
| 0 | loading the data in the dataframe | 1 |
| 1 | Counting of values in Target column | 1 |
| 2 | missing value analysis | 2 |
| 3 | Understanding data types | 1 |
| 4 | listing columns with Flag | 1 |
| 5 | categorize missingness severity | 2 |
| 6 | Does missingness differ between defaulter and repayers? | 2 |
| 7 | saving missingness in the document | 2 |
| 8 | Focus on most meaningful business numerical columns | 2 |
| 9 | import warnings *(setup, skip)* | — |
| 10 | Re-plot AMT_INCOME_TOTAL with log scale to handle skew | 2 |
| 11 | Investigate the known anomaly in DAYS_EMPLOYED | 2 |
| 12 | convert DAYS_BIRTH to age in years | 2 *(previews Ch. 3's AGE_YEARS feature)* |
| 13 | cat_cols = df.select_dtypes(...) | 1 |
| 14 | Calculate default rate for each category | 2 |
| 15 | calculating correlation of all numerical columns with Target | 2 |
| 16 | Visualization of data | 2 |
| 17 | Write EDA summary to docs folder | 2 *(this cell produces EDA_summary.md)* |
| 18 | print("Just checking in venv is working.") *(stray test cell, ignore)* | — |

### 02_feature_engineering.ipynb (61 cells) — Chapter 3

Cells 0–20 build the engineered ratio, missingness, anomaly, and date features (with several checkpoint/debugging cells interspersed — those are real, keep them, they show what actually goes wrong mid-build). Cells 21–49 handle encoding (binary, one-hot, ordinal, target encoding) plus verification. Cells 50–60 handle feature selection and saving the final list.

| Cell range | What happens | Chapter |
|---|---|---|
| 0 | import pandas *(setup)* | — |
| 1–4 | Ratio features (DEBT_TO_INCOME etc.), validation, DTI-band check | 3 |
| 5–12 | IS_MISSING flags, DAYS_EMPLOYED anomaly handling, checkpoints | 3 |
| 13–20 | Date conversions (AGE_YEARS, EMPLOYMENT_YEARS), document/contact flag scores, IS_YOUNG_BORROWER, correlation summary | 3 |
| 21–32 | Binary encoding, one-hot encoding, ordinal encoding + debugging | 3 |
| 33–36 | Target encoding (OCCUPATION_TYPE, ORGANIZATION_TYPE), CODE_GENDER encoded separately *(this is the fairness/ECOA decision point)* | 3 |
| 37–49 | Verification checks, column-count reconciliation *(debugging, real but not conceptually new)* | 3 |
| 50–54 | Define final feature set, drop zero-variance / near-constant / duplicate columns | 3 |
| 55, 58 | Consolidate drops and check feature importance *(a quick LightGBM run used purely as a feature-selection tool — your first real preview of Chapter 6's model)* | 3 → previews 6 |
| 56 | Sanitise feature names for LightGBM | 3 |
| 59–60 | Drop zero-importance features, save final feature list and processed data | 3 |

### 03_model_training.ipynb (16 cells) — Chapters 4–7

| Cell | Title | Chapter |
|---|---|---|
| 0 | Master import cell | — |
| 1 | Loading the processed data from Phase 3 | 4 |
| 2 | Load feature list and separate X, y | 4 |
| 3 | print(lines) *(checkpoint)* | — |
| 4 | Stratified train/validation/test split (60/20/20, stratify=y) | 4 |
| 5 | Save splits to disk | 4 |
| 6 | Impute missing values (SimpleImputer, median) | 4 |
| 7 | Feature scaling (StandardScaler) **+** Logistic Regression instantiation (`class_weight='balanced'`) | 4 & 5 |
| 8 | Generate predictions + AUC/Gini/AUC-PR for the baseline | 5 & 7 |
| 9 | Coefficient interpretation table (→ logreg_coefficients.csv) | 5 |
| 10 | Prepare data for LightGBM (scale_pos_weight calculated here) | 6 |
| 11 | LightGBM hyperparameters + training (lgb.train, early_stopping) | 6 |
| 12 | Full metric evaluation for LightGBM (AUC, Gini, AUC-PR, KS) | 7 |
| 13 | Definitive feature importance (→ feature_importance_final.csv) | 6 |
| 14 | Save the trained model | 6 |
| 15 | *(empty)* | — |

---

## How to actually use this while rehearsing

Open the chapter in the curriculum PDF/markdown and the matching notebook cell range side by side. Read the chapter's three sections first, attempt the "Try this yourself" prompt, then open the cells listed here and check your reasoning against the real code — not the other way around. For Chapter 3 especially, don't try to absorb all 61 cells in one sitting; the sub-ranges in the table above are natural stopping points (ratios → missingness/anomaly → date features → encoding → feature selection).

Chapters 8 and 9 have no cells to check against — that's expected, not something you're missing.
