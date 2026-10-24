# Software Support Ticket Resolution Time Estimation

This repository contains the prototype implementation for estimating software support ticket resolution times (`resolution_hours`). The analysis compares a naive baseline, Ordinary Least Squares (OLS) Linear Regression, Polynomial Regression (degrees 2 and 3), Ridge Regression, and Lasso Regression using a reproducible cross-validation and evaluation pipeline.

---

## 1. Project Overview & Tasks

1. **Data Inspection:** Tabular regression dataset loaded and verified for shape (240 rows, 6 columns), data types, and missing values.
2. **Feature Selection & Leakage Prevention:**
   - **Target ($y$):** `resolution_hours`.
   - **Excluded Identifier:** `ticket_id` (non-predictive row key).
   - **Excluded Leakage Risk:** `lines_changed` represents post-fix effort recorded only after ticket closure. Retaining it at initial triage time constitutes data leakage; it was excluded from predictors.
   - **Predictors ($X$):** `complexity_score`, `team_experience_years`, `reopened`.
3. **Data Splitting:** 80/20 train/test split with `random_state=42` ($N_{\text{train}}=192$, $N_{\text{test}}=48$).
4. **Baseline & Advanced Modeling:** Fit and compared `DummyRegressor(strategy='mean')`, `LinearRegression`, `PolynomialFeatures` (degrees 2 & 3), and scaled `Ridge` / `Lasso` regularized models with 5-fold cross-validation.
5. **Diagnostics:** Generated observed vs. predicted and residual scatter plots to evaluate model fit and heteroscedasticity.

---

## 2. Model Performance Summary

Evaluated on the held-out test set ($N_{\text{test}} = 48$):

| Model | Selection / Hyperparameters | Test MAE (hrs) | Test RMSE (hrs) | Test $R^2$ |
| :--- | :--- | :--- | :--- | :--- |
| **Dummy Regressor** | Mean baseline ($\bar{y}_{\text{train}} \approx 21.57$) | 9.1140 | 10.5977 | -0.0210 |
| **Linear Regression** | Standard OLS (`complexity`, `team_exp`, `reopened`) | **2.5905** | **3.2192** | **0.9058** |
| **Polynomial Reg. (deg=2)** | 5-Fold CV Train RMSE: 3.5118 | 2.7818 | 3.5528 | 0.8852 |
| **Polynomial Reg. (deg=3)** | 5-Fold CV Train RMSE: 3.6221 | 2.8233 | 3.6249 | 0.8805 |
| **Ridge Regression** | Scaled, $\alpha = 0.1207$ | 2.5911 | 3.2195 | 0.9058 |
| **Lasso Regression** | Scaled, $\alpha = 0.0010$ | 2.5906 | 3.2192 | 0.9058 |

---

## 3. Analysis Questions & Discussion

### Task 9: Recommendation
* **Selected Model:** Standard Linear Regression (or Ridge with low regularization penalty).
* **Supporting Evidence:** Linear regression matches top test performance (MAE $\approx 2.59$ hrs, RMSE $\approx 3.22$ hrs, $R^2 \approx 0.91$) while preserving clean interpretability. Cross-validation reveals that polynomial expansions overfit sample noise, worsening generalization error.
* **Additional Data / Pre-deployment Checks:** Collect categorical issue domains, ticket priority levels, and individual on-call schedules before practical deployment.

### Question 10: Why is the mean baseline useful even though it ignores all features?
The mean baseline sets an objective performance floor. It tests whether learning algorithms capture meaningful variance beyond the sample mean ($R^2 > 0$). A score near zero or negative immediately indicates that the model fails to extract predictive signal.

### Question 11: Which model has the lowest MAE? Which has the lowest RMSE? Are they the same, and why might they differ?
Standard Linear Regression and Lasso/Ridge achieve essentially identical best metrics (MAE $\approx 2.59$ hrs, RMSE $\approx 3.22$ hrs). MAE and RMSE can differ because MAE penalizes deviations linearly, whereas RMSE squares errors, disproportionately penalizing large outliers. A model with fewer massive errors may achieve a superior RMSE even if its median absolute deviation is comparable.

### Question 12: Did polynomial features improve validation performance? What is the evidence of possible overfitting?
No. Polynomial expansions degraded performance across both cross-validation and test evaluation:
- CV RMSE worsened from $3.18$ (Linear) to $3.51$ (Degree 2) and $3.62$ (Degree 3).
- Test $R^2$ dropped from $0.9058$ to $0.8805$.
The true underlying relationship is predominantly linear; higher-order interaction terms introduced collinearity and modeled training sample noise.

### Question 13: How did regularization affect performance and coefficient magnitudes?
Because the feature space is compact ($p = 3$) and devoid of strong multicollinearity, cross-validation selected minimal penalty values ($\alpha_{\text{Ridge}} \approx 0.1207$, $\alpha_{\text{Lasso}} \approx 0.0010$). Regularized models produced coefficient magnitudes that closely mirrored ordinary least squares:
- `complexity_score`: $+8.65$ (positive driver of duration)
- `team_experience_years`: $-1.16$ (efficiency gain)
- `reopened`: $+1.76$ (penalty for rework)

### Question 14: Why is scaling included in the Ridge/Lasso pipeline?
Regularization penalties ($L_2: \sum \beta_j^2$, $L_1: \sum |\beta_j|$) are scale-sensitive. If features have differing numerical ranges (e.g., complexity 1–10 vs. experience 0–15), unscaled regularization penalizes features with wider ranges more heavily, artificially driving their coefficients toward zero regardless of true predictive utility.

### Question 15: Identify one potential leakage issue in the dataset and explain how you handled it.
`lines_changed` is a target leakage risk. The line count modified during a fix is determined ex-post (after the ticket is resolved) and is unavailable when a ticket is opened or triaged. It was excluded from predictor matrix $X$.

### Question 16: Would a random split be appropriate if tickets were collected over several years or came from repeated customers/teams? Explain.
No:
- **Temporal Drift:** Software codebases, architectural patterns, and developer velocity shift over time. A time-based chronological split is required to evaluate historical generalization.
- **Group Correlation:** Tickets from repeated customers or teams share latent correlations. A grouped split strategy (`GroupKFold` on `team_id` or `client_id`) is necessary to prevent intra-cluster leakage across training and test splits.

### Question 17: What additional tests would you perform before deploying this model?
1. **Residual Heteroscedasticity Analysis:** Verify that residual variance remains uniform across all complexity tiers.
2. **Subgroup Disparity Audits:** Confirm error metrics do not systematically degrade for specific teams or specialized issue categories.
3. **Data Drift & Outlier Guardrails:** Test pipeline robustness against out-of-range complexity inputs or zero experience entries.
4. **Shadow Validation:** Run the model silently in production triage to validate estimates against actual resolution times before presenting predictions to engineering teams.

---

## 4. How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/ticket-resolution-ml.git
   cd ticket-resolution-ml
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch JupyterLab or execute the notebook:
   ```bash
   jupyter lab lab_ticket_resolution_walkthrough.ipynb
   ```