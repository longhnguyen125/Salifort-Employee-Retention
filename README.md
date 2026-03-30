# Salifort Motors — Employee Retention Analysis

---

## Overview
Salifort Motors is a fictional French-based alternative energy vehicle manufacturer.
This project analyzes HR data to predict which employees are likely to leave and 
identify the key factors driving turnover.

**Business Question:** *What factors are most likely to make an employee leave?*

---

## Dataset
- **File:** `HR_capstone_dataset.csv`
- **Size:** 14,999 rows × 10 columns (11,991 after removing duplicates)
- **Target:** `left` — 1 = left, 0 = stayed (17% attrition rate)

---

## PACE Framework

### 🟡 Plan
The target variable `left` is binary, making this a **classification problem**.
The goal is to build a model that predicts whether an employee will leave and 
surface the features driving that decision. Two model families were evaluated: 
Decision Tree and Random Forest. Success was measured by AUC, recall, and accuracy.

---

### 🔵 Analyze
**Data Cleaning:**
- Renamed columns for consistency, confirmed no missing values
- Removed 3,008 duplicate rows
- Identified 824 tenure outliers via IQR (bounds: 1.5 – 5.5 years)

**Key EDA Findings:**
- Every employee assigned **7 projects left** — 100% attrition at max load
- Two quitter groups: **overworked** (245–315 hrs/month, satisfaction ≈ 0) and 
  **underworked/bored** (~150 hrs/month, satisfaction ≈ 0.4)
- **4-year employees** had the lowest satisfaction of any tenure group
- Very few promotions despite widespread overwork
- High evaluation scores were almost exclusively given to **200+ hr/month workers**
- `satisfaction_level` had the strongest correlation with leaving **(r = −0.35)**

---

### 🟢 Construct
**Feature Engineering:**
- Encoded `salary` ordinally: low=0, medium=1, high=2
- One-hot encoded `department`
- **Round 2:** Dropped `satisfaction_level` (not always available in practice);
  replaced `average_monthly_hours` with binary `overworked` flag (>175 hrs = 1)

**Modeling:**
Both rounds trained Decision Tree and Random Forest using `GridSearchCV` 
with 4-fold cross-validation, optimizing for AUC.

| | Round 1 | Round 2 |
|---|---|---|
| Features | All features including satisfaction | No satisfaction; overworked flag instead |
| Purpose | Performance ceiling | Real-world deployable |

---

### 🔴 Execute
**Model Results:**

| Model | Recall | Accuracy | AUC |
|---|---|---|---|
| Decision Tree — R1 | 91.7% | 97.2% | 0.970 |
| Random Forest — R1 | 91.6% | 97.8% | 0.980 |
| Decision Tree — R2 | 90.4% | 95.9% | 0.959 |
| **★ Random Forest — R2** | **90.4%** | **96.2%** | **0.964** |

**Recommended model: Random Forest Round 2** — catches 90 out of every 100 
employees who will leave, without relying on satisfaction survey data.

**Top Predictors (both models agree):**

| Feature | Importance |
|---|---|
| `last_evaluation` | ~35% |
| `number_project` | ~34% |
| `tenure` | ~20% |
| `overworked` | ~10% |

**Recommendations:**
1. **Cap projects at 5** — all 7-project employees left
2. **Review employees at year 3–4** — dissatisfaction peaks at the 4-year mark
3. **Reward long hours or reduce workload** — most employees work above the 166 hr baseline
4. **Reform evaluations** — high scores should not require 200+ hrs/month
5. **Clarify overtime policies** — ambiguity drives dissatisfaction
6. **Hold culture discussions** — burnout is systemic, not individual

---

## Repository Structure
```
├── HR_capstone_dataset.csv                  # Raw dataset
├── HR_Project.ipynb                         # Analysis notebook
├── Salifort_Motors_Executive_Summary.pptx   # Executive summary deck
└── README.md
```

---

## Tools
Python, pandas, numpy, matplotlib, seaborn, scikit-learn, Jupyter Notebook

---

*Google Advanced Data Analytics Certificate — Course 7 Capstone | 2026*
