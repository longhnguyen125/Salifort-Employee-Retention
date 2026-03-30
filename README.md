# Salifort Motors — Employee Retention Analysis
### Google Advanced Data Analytics Certificate | Course 7 Capstone Project

---

## Project Overview
Salifort Motors is a fictional French-based alternative energy vehicle manufacturer 
with a global workforce of over 100,000 employees. This project analyzes HR data to 
predict employee turnover and identify the key factors driving it, using tree-based 
machine learning models.

**Business Question:** *What factors are most likely to make an employee leave the company?*

---

## Dataset
- **Source:** HR_capstone_dataset.csv
- **Size:** 14,999 rows × 10 columns (11,991 after removing duplicates)
- **Target variable:** `left` (1 = left, 0 = stayed)

| Feature | Description |
|---|---|
| `satisfaction_level` | Employee self-reported satisfaction (0–1) |
| `last_evaluation` | Most recent performance evaluation score (0–1) |
| `number_project` | Number of projects assigned |
| `average_monthly_hours` | Average hours worked per month |
| `tenure` | Years at the company |
| `work_accident` | Whether the employee had a work accident |
| `promotion_last_5years` | Whether promoted in last 5 years |
| `department` | Department name |
| `salary` | Salary level (low / medium / high) |

---

## PACE Framework

This project follows the **PACE** framework — a structured approach to data analytics
consisting of four stages: **Plan, Analyze, Construct, and Execute.**

---

### 🟡 P — Plan

The Plan stage defines the problem, identifies stakeholders, and outlines the 
approach before any data work begins.

**Business context:**
Salifort Motors is experiencing unexpected employee turnover. The senior leadership 
team wants to understand why employees are leaving, whether attrition is predictable, 
and which groups are most at risk. Retaining employees is critical — replacing a 
single employee is costly in time, resources, and institutional knowledge.

**Stakeholders:**
- Senior leadership team (primary decision makers)
- HR department (will act on model outputs)
- Department managers (affected by turnover patterns)

**Analytical approach:**
Since the target variable `left` is binary (0 or 1), this is a **classification 
problem**. The team evaluated two model families:
- Logistic Regression (baseline)
- Tree-based models: Decision Tree and Random Forest *(focus of this project)*

**Success criteria:**
- Build a model that reliably predicts whether an employee will leave
- Identify the top features driving turnover
- Deliver actionable recommendations to reduce attrition

---

### 🔵 A — Analyze

The Analyze stage covers data collection, cleaning, and exploratory data analysis 
(EDA) to understand the dataset before modeling.

**Data Cleaning:**
- Renamed columns for consistency (`Work_accident` → `work_accident`, 
  `time_spend_company` → `tenure`, etc.)
- Found **no missing values** across all 10 columns
- Identified and removed **3,008 duplicate rows**, leaving 11,991 clean records
- Detected **824 outliers** in `tenure` using the IQR method 
  (lower limit: 1.5 years, upper limit: 5.5 years)

**Key EDA Findings:**

**Workload & Hours:**
- **17%** of employees left the company — roughly 1 in 6 workers
- Every employee assigned **7 projects left** — 100% attrition rate at maximum load
- The standard monthly working hours baseline is **166.67 hrs** 
  (40 hrs/week × 50 weeks ÷ 12 months). Nearly all employees — even those who 
  stayed — worked significantly above this
- Two distinct quitter groups emerged from the data:
  - **Group 1 (Overworked):** 245–315 hrs/month, satisfaction score near 0 — 
    likely burned out and exhausted
  - **Group 2 (Underworked):** ~150 hrs/month, satisfaction score ~0.4 — 
    likely bored, underutilized, or disengaged

**Tenure & Satisfaction:**
- Employees at the **4-year mark** showed the lowest satisfaction of any tenure group,
  suggesting a policy change or unaddressed cultural shift at that milestone
- Employees with **6+ years of tenure** rarely left — they appeared to eventually 
  find their footing or settle into the company
- Employees who left had a mean satisfaction score of **0.44 vs 0.67** for those who
  stayed — a 34% gap
- `satisfaction_level` had the strongest single correlation with leaving **(r = −0.35)**

**Promotions & Evaluations:**
- Very few employees received promotions in the last 5 years, despite widespread 
  overwork
- High evaluation scores were almost exclusively given to employees working **200+ 
  hours per month** — an unfair and unsustainable standard
- A strong positive correlation exists between hours worked and evaluation scores, 
  suggesting effort is being measured by time, not contribution

**Correlation Heatmap:**
- `number_project`, `average_monthly_hours`, and `last_evaluation` all positively 
  correlate with each other
- `left` negatively correlates most strongly with `satisfaction_level`

---

### 🟢 C — Construct

The Construct stage covers feature engineering, encoding, model building, and 
hyperparameter tuning.

**Feature Engineering & Encoding:**
- Encoded `salary` as an ordinal variable: `low = 0`, `medium = 1`, `high = 2`
- One-hot encoded the `department` column using `pd.get_dummies()`
- **Round 2 feature engineering:** Created a binary `overworked` column 
  (`1` if average monthly hours > 175, else `0`) to replace the raw hours column 
  with a more interpretable signal

**Two Modeling Rounds:**

The project was run in two rounds to test model performance under different 
feature conditions:

| | Round 1 | Round 2 |
|---|---|---|
| **Key difference** | Used all features including `satisfaction_level` | Dropped `satisfaction_level`; added binary `overworked` flag |
| **Rationale** | Establish a performance ceiling | Simulate real-world deployment where satisfaction data may not be available |
| **Dataset used** | Full `df_enc` (11,991 rows, 18 features) | `df2` (11,991 rows, 17 features) |
| **Train/Test split** | 75% / 25%, stratified | 75% / 25%, stratified |

**Models Built:**
Both rounds trained and tuned a **Decision Tree** and a **Random Forest** using 
`GridSearchCV` with 4-fold cross-validation, optimizing for **AUC (ROC)**.

Decision Tree hyperparameters searched:
```python
cv_params = {
    'max_depth': [4, 6, 8, None],
    'min_samples_leaf': [2, 5, 1],
    'min_samples_split': [2, 4, 6]
}
```

Random Forest hyperparameters searched:
```python
cv_params = {
    'max_depth': [3, 5, None],
    'max_features': [1.0],
    'max_samples': [0.7, 1.0],
    'min_samples_leaf': [1, 2, 3],
    'min_samples_split': [2, 3, 4],
    'n_estimators': [300, 500]
}
```

**Best Parameters Found:**

| Model | Best Parameters |
|---|---|
| Decision Tree R1 | max_depth=4, min_samples_leaf=5, min_samples_split=2 |
| Random Forest R1 | max_depth=5, max_samples=0.7, min_samples_split=4, n_estimators=500 |
| Decision Tree R2 | max_depth=6, min_samples_leaf=2, min_samples_split=6 |
| Random Forest R2 | max_depth=5, max_samples=0.7, min_samples_split=2, n_estimators=300 |

---

### 🔴 E — Execute

The Execute stage covers model evaluation, interpretation of results, and delivery 
of business recommendations.

**Model Performance:**

| Model | Precision | Recall | F1 | Accuracy | AUC |
|---|---|---|---|---|---|
| Decision Tree — Round 1 | 91.5% | 91.7% | 91.6% | 97.2% | 0.970 |
| Random Forest — Round 1 | 95.0% | 91.6% | 93.2% | 97.8% | 0.980 |
| Decision Tree — Round 2 | 85.7% | 90.4% | 87.9% | 95.9% | 0.959 |
| **★ Random Forest — Round 2** | **87.0%** | **90.4%** | **88.7%** | **96.2%** | **0.964** |

**Recommended model: Random Forest — Round 2**

Although Round 1 achieved slightly higher scores, Round 2 is the recommended model 
for deployment because it does not rely on `satisfaction_level` — a metric that 
requires active employee surveys and may not always be available in a real HR system. 
Round 2 replaces it with `overworked`, a feature that can be derived directly from 
payroll data.

**Confusion Matrix (Random Forest Round 2 — Test Set, n=2,998):**

|  | Predicted: Stayed | Predicted: Left |
|---|---|---|
| **Actual: Stayed** | 2,433 ✅ | 67 ⚠️ |
| **Actual: Left** | 48 ❌ | 450 ✅ |

- **2,433** correctly predicted as staying
- **450** correctly predicted as leaving
- **67** false alarms (predicted to leave, but stayed)
- **48** missed — employees who left that the model did not catch

In plain terms: out of every 100 employees who actually quit, **the model catches 90 
of them in advance**, giving HR a meaningful window to intervene.

**Feature Importance (Top 4 — consistent across both models):**

| Rank | Feature | Importance |
|---|---|---|
| 1 | `last_evaluation` | ~35% |
| 2 | `number_project` | ~34% |
| 3 | `tenure` | ~20% |
| 4 | `overworked` | ~10% |

> Department, salary, and work accidents contributed less than 1% each to 
> the model's predictions.

**Key Findings:**
- Employees are likely leaving due to **poor management and burnout**
- Leaving is tied to **longer working hours, high project counts, and lower 
  satisfaction levels**
- Employees with **6+ years tenure tend to stay**; dissatisfaction peaks around 
  the **4-year mark**
- **High evaluation scores** are disproportionately tied to excessive working hours

**Recommendations:**

| # | Action | Why |
|---|---|---|
| 1 | **Cap projects at 5 per employee** | All 7-project employees left; 6-project attrition was also elevated |
| 2 | **Promote or review employees at year 3–4** | 4-year employees show the lowest satisfaction of any group |
| 3 | **Reward long hours or reduce workload expectations** | Most employees work well above the 166 hr/month baseline |
| 4 | **Reform evaluation criteria** | High scores should not be reserved for 200+ hr/month workers only |
| 5 | **Clarify overtime and time-off policies** | Ambiguity around expectations is a known driver of dissatisfaction |
| 6 | **Hold company-wide culture discussions** | Burnout is a systemic problem — not an individual one |

---

## Repository Structure
```
├── HR_capstone_dataset.csv                    # Raw dataset
├── HR_Project.ipynb                           # Main analysis notebook
├── Salifort_Motors_Executive_Summary.pptx     # Executive summary deck
└── README.md
```

---

## Tools & Libraries

- **Python** — pandas, numpy, matplotlib, seaborn
- **Scikit-learn** — DecisionTreeClassifier, RandomForestClassifier, GridSearchCV
- **Jupyter Notebook**

