# Credit Scoring Engine with Explainability Dashboard
### Product Requirements Document (PRD)
**Version:** 1.0 — Draft &nbsp;|&nbsp; **Date:** February 2026 &nbsp;|&nbsp; **Status:** PRD Complete — EDA Phase Upcoming

---

## Project Metadata

| Field | Detail |
|---|---|
| Project Name | Credit Scoring Engine with Explainability Dashboard |
| Document Type | Product Requirements Document (PRD) |
| Version | 1.0 — Draft |
| Date | February 2026 |
| Dataset | Home Credit Default Risk — `application_train.csv` |
| Data Size | ~307,000 rows × 122 columns |
| Primary Stack | Python, LightGBM, SHAP, Streamlit, scikit-learn |
| Repo | `github.com/[your-username]/credit-scoring-engine` |
| Status | PRD Complete — EDA Phase Upcoming |

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Goals & Success Metrics](#3-goals--success-metrics)
4. [Scope](#4-scope)
5. [User Personas](#5-user-personas)
6. [Functional Requirements](#6-functional-requirements)
7. [Non-Functional Requirements](#7-non-functional-requirements)
8. [Dataset & Data Requirements](#8-dataset--data-requirements)
9. [Technology Stack](#9-technology-stack)
10. [Project Folder Structure](#10-project-folder-structure)
11. [Project Phases & Milestones](#11-project-phases--milestones)
12. [Risks & Mitigations](#12-risks--mitigations)
13. [Known Limitations](#13-known-limitations)
14. [Appendix — Glossary](#14-appendix--glossary)

---

## 1. Executive Summary

The **Credit Scoring Engine with Explainability Dashboard** is a portfolio-grade machine learning application that predicts the probability of a loan applicant defaulting on their credit obligations. The system is built on the **Home Credit Default Risk** dataset from Kaggle — a real-world dataset containing approximately 307,000 applicants and 122 features.

Beyond simple prediction, the system includes a **Streamlit-powered interactive dashboard** that provides SHAP-based explainability, enabling end users to understand *why* a particular decision was made for each applicant. This directly addresses a critical gap in traditional credit scoring: the black-box nature of model decisions.

This project demonstrates the full ML product lifecycle — from raw data ingestion and exploratory analysis, through model development and evaluation, to a deployed, user-facing application with explainability built in. It is designed to serve as a flagship portfolio artifact showcasing both data science depth and product thinking.

---

## 2. Problem Statement

Millions of people struggle to obtain loans due to insufficient or non-existent credit histories. Traditional credit scoring systems rely heavily on credit bureau data, which systematically excludes underbanked populations. Home Credit Group attempts to close this gap by using alternative data to assess creditworthiness — but doing so responsibly requires accurate, explainable predictions.

The core problems this project addresses are:

- **Binary classification accuracy:** Can we reliably predict whether an applicant will default?
- **Class imbalance:** Default events are rare (~8% of the dataset), making standard models unreliable without intervention.
- **Explainability gap:** Loan officers and regulators need to understand *why* a model made a decision, not just *what* the decision was.
- **Portfolio gap:** The developer needs a real-world, end-to-end ML project that demonstrates PM thinking, engineering depth, and deployment skills.

---

## 3. Goals & Success Metrics

### 3.1 Primary Goals

- Train a high-performance binary classification model using **LightGBM** on the Home Credit dataset
- Handle class imbalance using **imbalanced-learn** techniques (SMOTE and/or class weighting)
- Implement **SHAP-based explainability** for individual and global model interpretations
- Build a **Streamlit dashboard** that allows non-technical users to interact with predictions and explanations
- Produce a clean, well-documented **GitHub repository** suitable for portfolio presentation

### 3.2 Success Metrics

| Metric | Target | Rationale |
|---|---|---|
| AUC-ROC | >= 0.75 | Primary competition metric; measures ranking ability across thresholds |
| Recall (Default class) | >= 0.70 | Critical — missing a default is more costly than a false alarm |
| Precision | >= 0.60 | Balance against recall to avoid excessive false positives |
| F1-Score | >= 0.65 | Harmonic mean; useful given class imbalance |
| Dashboard load time | < 5 seconds | Acceptable UX for a local Streamlit app |
| SHAP waterfall render | < 3 seconds per applicant | Interactive explainability must feel responsive |
| GitHub repo quality | README, docstrings, clean commits | Portfolio artifact standard |

---

## 4. Scope

### 4.1 In Scope — Phase 1 (This Project)

- Ingestion and preprocessing of `application_train.csv`
- Exploratory Data Analysis (EDA): distributions, correlations, missing value analysis, class imbalance assessment
- Feature engineering: encoding, scaling, imputation, and derived features
- Model training: LightGBM as primary model; Logistic Regression as interpretable baseline
- Handling class imbalance via SMOTE and/or LightGBM `class_weight` parameter
- Model evaluation: AUC-ROC, F1, Precision, Recall, confusion matrix
- SHAP explainability: global feature importance + per-applicant waterfall charts
- Streamlit dashboard: applicant input form, prediction output, SHAP visualization
- Documentation: this PRD, EDA report, `README.md`, inline code comments

### 4.2 Out of Scope — Potential Phase 2

- Integration of supplementary tables (`bureau.csv`, `previous_application.csv`, etc.)
- FastAPI REST endpoint and productionization
- Cloud deployment (AWS / GCP / Azure)
- Real-time scoring pipeline
- Model monitoring and drift detection
- Multi-language or mobile support

---

## 5. User Personas

| Attribute | Loan Officer | Risk Analyst | Portfolio Reviewer (Recruiter/PM) |
|---|---|---|---|
| Primary Goal | Assess applicant default risk quickly and confidently | Understand model behavior and fairness across segments | Evaluate developer's ML and product thinking skills |
| Pain Points | Black-box models provide no explanation; hard to justify decisions | No visibility into which features drive risk scores | Generic ML projects with no real-world complexity or UI |
| Tech Savviness | Low-to-moderate; comfortable with web forms | High; comfortable with data and charts | Moderate; looking for storytelling and structure |
| Key Need | Clear Default / Non-Default output with plain-language explanation | Global SHAP plots, feature importance, segment analysis | Clean repo, PRD, EDA notebook, working dashboard |

---

## 6. Functional Requirements

### 6.1 Data Pipeline

- **FR-D1:** The system shall load `application_train.csv` from the `data/` directory
- **FR-D2:** The system shall perform automatic missing value imputation (median for numeric; mode for categorical)
- **FR-D3:** The system shall encode categorical variables using label encoding or one-hot encoding as appropriate
- **FR-D4:** The system shall detect and report class distribution of the `TARGET` column
- **FR-D5:** The system shall apply SMOTE or class weighting to address class imbalance during training
- **FR-D6:** The pipeline shall be reproducible via a single script or notebook execution

### 6.2 Model Training & Evaluation

- **FR-M1:** The system shall train a LightGBM binary classifier as the primary model
- **FR-M2:** The system shall train a Logistic Regression model as an interpretable baseline
- **FR-M3:** The system shall split data into 70% train / 15% validation / 15% test sets with a fixed `random_state=42`
- **FR-M4:** The system shall perform hyperparameter tuning on LightGBM (`num_leaves`, `learning_rate`, `n_estimators`, `min_child_samples`)
- **FR-M5:** The system shall report AUC-ROC, F1-Score, Precision, Recall, and a confusion matrix for both models
- **FR-M6:** The trained model shall be serialized to the `models/` directory using `joblib` or `pickle`

### 6.3 Explainability

- **FR-E1:** The system shall compute SHAP values for the LightGBM model using the `shap` library
- **FR-E2:** The system shall generate a global SHAP summary plot (beeswarm) showing top 20 features
- **FR-E3:** The system shall generate a per-applicant SHAP waterfall chart explaining individual predictions
- **FR-E4:** SHAP output shall be accessible within the Streamlit dashboard

### 6.4 Streamlit Dashboard

- **FR-S1:** The dashboard shall display a sidebar form for entering or selecting applicant features
- **FR-S2:** The dashboard shall display a real-time prediction: **Default Risk: HIGH / LOW** with a probability score
- **FR-S3:** The dashboard shall display the top 5 contributing features for the selected applicant via a SHAP waterfall chart
- **FR-S4:** The dashboard shall include a global model overview tab with feature importance and AUC-ROC curve
- **FR-S5:** The dashboard shall be launchable locally with: `streamlit run app/main.py`

---

## 7. Non-Functional Requirements

| ID | Category | Requirement |
|---|---|---|
| NFR-1 | Performance | Single applicant prediction + SHAP computation must complete in under 3 seconds on a standard laptop |
| NFR-2 | Reproducibility | All results must be reproducible with a fixed `random_state=42` seed throughout the pipeline |
| NFR-3 | Portability | All dependencies must be installable via `pip install -r requirements.txt` |
| NFR-4 | Code Quality | All `src/` modules must include docstrings; notebooks must have markdown explanations between cells |
| NFR-5 | Version Control | All commits must be meaningful and descriptive; `data/` and `models/` must be in `.gitignore` |
| NFR-6 | Portfolio Readiness | `README.md` must include: project summary, setup instructions, architecture diagram, and sample screenshots |

---

## 8. Dataset & Data Requirements

### 8.1 Primary Dataset

| Attribute | Detail |
|---|---|
| Source | Kaggle — Home Credit Default Risk Competition |
| File | `application_train.csv` |
| Rows | ~307,511 applicants |
| Columns | 122 features + 1 target column (`TARGET`) |
| Target Variable | `TARGET`: 1 = Defaulted, 0 = Did Not Default |
| Class Distribution | ~91.9% Non-Default / ~8.1% Default (significant imbalance) |
| Feature Types | Numeric (continuous & discrete), Categorical (binary & multi-class), Flags |
| Known Issues | Missing values present across many columns; some columns have >40% missing |
| Storage Location | `data/` directory (excluded from Git via `.gitignore`) |

### 8.2 Data Split Strategy

| Split | Proportion | Approximate Size | Purpose |
|---|---|---|---|
| Training Set | 70% | ~215,000 rows | Model training + SMOTE application |
| Validation Set | 15% | ~46,000 rows | Hyperparameter tuning |
| Test Set | 15% | ~46,000 rows | Final unbiased evaluation |

### 8.3 Feature Engineering Plan

- Drop columns with > 50% missing values after EDA review
- Impute remaining missing numerics with column median
- Impute remaining missing categoricals with column mode
- Label-encode binary categoricals (Y/N, M/F, etc.)
- One-hot encode multi-class categoricals with low cardinality
- Derive new features if EDA reveals useful signal (e.g., `CREDIT_INCOME_RATIO = AMT_CREDIT / AMT_INCOME_TOTAL`)

---

## 9. Technology Stack

| Layer | Technology | Version | Purpose |
|---|---|---|---|
| Language | Python | 3.8+ | Core development language |
| Data Processing | pandas, numpy | Latest stable | Data manipulation and numerical computation |
| Visualization / EDA | matplotlib, plotly, seaborn | Latest stable | Charts for EDA and dashboard |
| ML — Baseline | scikit-learn (Logistic Regression) | Latest stable | Interpretable baseline model and preprocessing utilities |
| ML — Primary Model | LightGBM | Latest stable | High-performance gradient boosting for tabular data |
| Class Imbalance | imbalanced-learn (SMOTE) | Latest stable | Synthetic minority oversampling |
| Explainability | shap | Latest stable | SHAP values for global and local model explanation |
| Dashboard | Streamlit | Latest stable | Interactive front-end for predictions and SHAP visualization |
| API Layer (Phase 2) | FastAPI + uvicorn | Latest stable | REST endpoint for model serving |
| Model Persistence | joblib | Latest stable | Serialize and load trained model artifacts |
| Version Control | Git + GitHub | — | Source control and portfolio hosting |
| Experiment Tracking | MLflow (optional) | Latest stable | Log model runs, params, and metrics |

---

## 10. Project Folder Structure

```
credit-scoring-engine/
│
├── data/                  # Raw data (gitignored)
├── notebooks/             # EDA and model development notebooks
├── src/                   # Reusable Python modules
│   ├── preprocessing.py   # Data cleaning and feature engineering
│   ├── train.py           # Model training pipeline
│   └── evaluate.py        # Metrics and evaluation utilities
├── app/                   # Streamlit dashboard
│   └── main.py            # Dashboard entry point
├── models/                # Saved model artifacts (gitignored)
├── docs/                  # PRD and project notes
│   └── PRD.md             # This file
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 11. Project Phases & Milestones

| # | Phase | Key Deliverables | Status |
|---|---|---|---|
| 1 | Environment Setup | Virtual env, libraries installed, folder structure, GitHub repo created | ✅ Complete |
| 2 | PRD | This document (`docs/PRD.docx` + `docs/PRD.md`) | ✅ Complete |
| 3 | Exploratory Data Analysis | EDA notebook, missing value report, class imbalance analysis, correlation heatmap | 🔲 Upcoming |
| 4 | Feature Engineering | Cleaned dataset, encoded features, derived columns, preprocessing pipeline script | 🔲 Upcoming |
| 5 | Model Development | Trained LightGBM + baseline, hyperparameter tuning, evaluation report, model artifact saved | 🔲 Upcoming |
| 6 | Explainability Layer | SHAP summary plot, per-applicant waterfall charts, SHAP integration tested | 🔲 Upcoming |
| 7 | Streamlit Dashboard | Working app with prediction form, SHAP charts, global metrics tab | 🔲 Upcoming |
| 8 | Documentation & Polish | Final README with screenshots, clean commit history, `requirements.txt` verified | 🔲 Upcoming |

---

## 12. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation Strategy |
|---|---|---|---|
| Severe class imbalance (~8% default rate) degrades model recall | High | High | Apply SMOTE on training set; use `scale_pos_weight` in LightGBM; tune classification threshold |
| High proportion of missing values in key columns | High | Medium | Drop cols with >50% missing after EDA; impute the rest; document decisions in EDA notebook |
| LightGBM overfits on training data | Medium | High | Use early stopping on validation set; tune `num_leaves`, `min_child_samples`, regularization params |
| SHAP computation too slow for interactive dashboard | Medium | Medium | Precompute SHAP values on test set; use `TreeExplainer` (fast) rather than `KernelExplainer` |
| GitHub repo doesn't read as professional | Low | High | Write README first; maintain clean commits from day one; add architecture diagram and screenshots |

---

## 13. Known Limitations

- **Supplementary tables deferred:** Phase 1 uses only `application_train.csv`. The competition provides several supplementary tables (`bureau.csv`, `previous_application.csv`, `installments_payments.csv`, etc.) that could significantly improve model performance. These are deferred to Phase 2.
- **Dataset specificity:** The model is trained on historical data from Home Credit's specific customer base and may not generalize to other financial institutions without retraining.
- **Not production-hardened:** The Streamlit dashboard is designed for local execution. Authentication, rate limiting, and logging are out of scope for Phase 1.
- **Regulatory disclaimer:** SHAP values explain the model's behavior but do not constitute legal justification for credit decisions under regulations such as GDPR's right to explanation or ECOA in the US.

---

## 14. Appendix — Glossary

| Term | Definition |
|---|---|
| AUC-ROC | Area Under the Receiver Operating Characteristic Curve. Values range from 0.5 (random) to 1.0 (perfect). The primary evaluation metric for this project. |
| SHAP | SHapley Additive exPlanations. A game-theoretic approach to explaining individual model predictions by attributing a contribution value to each feature. |
| LightGBM | Light Gradient Boosting Machine. A fast, distributed gradient boosting framework optimized for tabular data. |
| SMOTE | Synthetic Minority Oversampling Technique. Generates synthetic samples of the minority class to address class imbalance. |
| Class Imbalance | A condition where one class (Default) is significantly underrepresented compared to another (Non-Default) in the training data. |
| Streamlit | An open-source Python framework for building interactive web applications for data science and machine learning. |
| Waterfall Chart | A SHAP visualization showing how each feature pushes the prediction higher or lower from the model's base value for a single applicant. |
| Precision | Of all applicants predicted as defaulters, the fraction who actually defaulted. TP / (TP + FP). |
| Recall | Of all actual defaulters, the fraction correctly identified. TP / (TP + FN). Prioritized over precision in this project. |
| F1-Score | Harmonic mean of Precision and Recall. Useful when class distribution is uneven. |

---

*This document is part of the Credit Scoring Engine portfolio project. Last updated: February 2026.*
