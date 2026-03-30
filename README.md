# Salifort Motors — Employee Retention Analysis

---

## Overview
Salifort Motors is a fictional French-based alternative energy vehicle manufacturer.
This project analyzes HR data to predict which employees are likely to leave and 
identify the key factors driving turnover.

---

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
surface the features driving that decision. Success was measured by AUC, recall,
and accuracy.

---

### 🔵 Analyze
**Data Cleaning:**
- Renamed columns for consistency, confirmed no missing values
- Removed 3,008 duplicate rows
- Identified 824 tenure outliers via IQR (bounds: 1.5 – 5.5 years)

**Key EDA Findings:**

**Workload & Hours**
- Roughly **17% of employees left** — about 1 in 6 workers
- Every employee assigned **7 projects left** — 100% attrition at max load
- The standard monthly baseline is 166.67 hrs. Nearly all employees worked
  significantly above this — overwork is the norm, not the exception
- Two distinct quitter groups identified:
  - **Overworked group:** 245–315 hrs/month with near-zero satisfaction — burned out
  - **Underworked group:** ~150 hrs/month with satisfaction ~0.4 — bored or disengaged

**Tenure & Satisfaction**
- Employees who left had a mean satisfaction score of **0.44 vs 0.67** for those
  who stayed — a clear and significant gap
- **4-year employees** showed the lowest satisfaction of any tenure group,
  suggesting an unaddressed policy or culture issue at that milestone
- Employees with **6+ years tenure rarely left** — they eventually found their footing

**Promotions & Evaluations**
- Very few employees were promoted in the last 5 years, despite widespread overwork
- High evaluation scores were almost exclusively given to **200+ hr/month workers** —
  effort is being measured by hours, not contribution
- A strong correlation exists between hours worked and evaluation score,
  creating an unfair and unsustainable standard

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
**Recommended model: Random Forest Round 2** — does not rely on satisfaction
survey data, making it practical for real HR deployment.

**Key Findings:**
- Employees are leaving due to **burnout and poor management**, not personal choice
- The strongest predictors of turnover are **evaluation score, project count,
  tenure, and being overworked** — department and salary barely matter
- Workload and performance standards are the root cause, not the symptom

**Recommendations:**
1. **Cap projects at 5** — all 7-project employees left; overload is the clearest signal
2. **Review employees at year 3–4** — dissatisfaction peaks at the 4-year mark
3. **Reward long hours or reduce workload** — most employees work above the 166 hr baseline
4. **Reform evaluations** — high scores should not require 200+ hrs/month
5. **Clarify overtime policies** — ambiguity around expectations drives dissatisfaction
6. **Hold culture discussions** — burnout is a systemic problem, not an individual one

---

## Tools
Python, pandas, numpy, matplotlib, seaborn, scikit-learn

