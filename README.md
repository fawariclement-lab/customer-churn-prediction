# Customer Churn Prediction Analysis

Predicting which e-commerce customers are about to churn (stop buying), so the business can intervene before they leave — built with the exact workflow in `churn_prediction.ipynb` (executed end-to-end, outputs saved in the notebook).

**Dataset:** `ecommerce_churn.csv` — 1,896 customers (948 churned / 948 stayed, perfectly balanced). Features: Tenure, Gender, SatisfactionScore, MaritalStatus, NumberOfAddress, Complain, OrderCount, DaySinceLastOrder.

## Method (as in the notebook)

1. **Exploration** — head/info/missing-value counts/Churn distribution/describe.
2. **Preprocessing** — dropped rows with missing `Churn` (none in this file); median-imputed `Tenure`, `SatisfactionScore`, `NumberOfAddress`, `OrderCount`, `DaySinceLastOrder` (287 missing cells total, mostly Tenure); one-hot encoded `Gender` + `MaritalStatus` with `drop_first=True`. Features = 9 predictors (CustomerID excluded).
3. **Split & scale** — 80/20 split (`random_state=42`); `StandardScaler` fit on the **training set only**, then applied to test — scaling after the split avoids data leakage (the template mentioned scaling in two places; this is the defensible reading).
4. **Models** — Logistic Regression (`max_iter=1000`) and Decision Trees at max_depth 3, 4, 5.
5. **Selection rule** — **Recall is the primary metric** (a missed churner costs more than a false alarm), checked against the train–test overfitting guide (<0.02 small, 0.02–0.05 moderate, >0.05 possible overfitting).

## Results

| Model | Train Acc | Train Prec | Train Recall | Test Acc | Test Prec | Test Recall |
|---|---|---|---|---|---|---|
| **Logistic Regression** | 0.814 | 0.802 | 0.828 | 0.805 | 0.794 | **0.850** |
| Decision Tree (depth=3) | 0.802 | 0.819 | 0.767 | 0.813 | 0.838 | 0.800 |
| Decision Tree (depth=4) | 0.823 | 0.804 | 0.846 | 0.797 | 0.786 | 0.845 |
| Decision Tree (depth=5) | 0.842 | 0.865 | 0.805 | 0.803 | 0.845 | 0.765 |

## Best model: Logistic Regression

- **Highest test Recall (0.850)** — on the 380-customer test set it caught **170 of 200 actual churners** (30 missed) with 44 false alarms (confusion matrix: TN 136, FP 44, FN 30, TP 170).
- **Best generalization** — train–test accuracy gap is only +0.009 and test recall (−0.023 gap) is actually *higher* than train recall: no overfitting by the guide.
- The depth-4 tree is a close second on recall (0.845) but with a moderate accuracy gap (+0.025). The depth-5 tree shows the classic overfitting pattern — train metrics climb while test recall falls to 0.765 (≈+0.04 recall gap).
- **Why it wins:** the churn signal here is mostly additive — each risk factor adds its own push — which is exactly what logistic regression models; the deeper trees chased interactions that were noise.

**Top churn drivers** (logistic coefficients on scaled features): Tenure **−1.59** (protective — long-tenure customers rarely leave), Complain **+0.82**, NumberOfAddress **+0.50**, OrderCount **+0.45**, DaySinceLastOrder **−0.43**.

## Recommendations

1. **Score customers monthly** with the logistic model and route the highest-risk segment to retention outreach while there's still time to act.
2. **Treat every complaint as a churn alert** — follow up within days with a goodwill gesture; complainers are the highest-risk identifiable group.
3. **Invest in the first 90 days** — short tenure is the #1 risk factor, so onboarding check-ins and a second-order incentive protect the most vulnerable cohort.
4. **Accept the false alarms** — at 0.794 precision, ~1 in 5 flags is a customer who would have stayed anyway; cheap insurance next to catching 85% of true churners.

## Files

- `churn_prediction.ipynb` — the full analysis (markdown narration + beginner-readable comments throughout, executed outputs included)
- `ecommerce_churn.csv` — source data
- `.venv/` — local Python environment (pandas, scikit-learn, seaborn, jupyter) used to run the notebook
