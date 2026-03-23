# 📈 Multiple Linear Regression — Economic Index Price Prediction

> Predicting stock market index price using macroeconomic indicators with Python & Scikit-learn

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-orange?logo=scikit-learn)
![Statsmodels](https://img.shields.io/badge/Statsmodels-0.14-green)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Project Overview

This project builds a **Multiple Linear Regression** model to predict the **Stock Index Price** using two macroeconomic indicators:

- 📉 **Interest Rate** — the central bank lending rate
- 👷 **Unemployment Rate** — percentage of the unemployed population

This is the **second project** in my ML learning series, extending Simple Linear Regression to multiple features. The focus is not just on building the model, but on understanding *why* every step is done — from scaling to residual analysis.

---

## 🗂️ Dataset

| Property | Value |
|----------|-------|
| Source | Custom economic dataset |
| Period | January 2016 – December 2017 |
| Rows | 24 monthly observations |
| Features | `interest_rate`, `unemployment_rate` |
| Target | `index_price` |

---

## 🛠️ Tech Stack

| Library | Purpose |
|---------|---------|
| `pandas` | Data loading & manipulation |
| `numpy` | Numerical operations |
| `matplotlib` | Base plotting |
| `seaborn` | Statistical visualizations |
| `scikit-learn` | Model building, scaling, evaluation |
| `statsmodels` | OLS summary with p-values & confidence intervals |

---

## 📁 Project Structure

```
multiple-linear-regression/
│
├── multiple_linear_regression_project.ipynb   # Main notebook (fully documented)
├── economic_index.csv                         # Dataset
└── README.md                                  # This file
```

---

## 🔄 Workflow

```
Load Data → EDA → Preprocess → Split → Scale → Train → Cross-Validate → Evaluate → Diagnose
```

### Steps in detail:

1. **Load & Explore** — shape, dtypes, missing values, descriptive stats
2. **Preprocessing** — drop irrelevant columns (`year`, `month`, index artifact)
3. **EDA** — pairplot, correlation matrix, regression plots
4. **Feature/Target Split** — X (features) vs y (target)
5. **Train-Test Split** — 75% train, 25% test, `random_state=42`
6. **StandardScaler** — fit on train only to prevent data leakage
7. **LinearRegression (sklearn)** — train the model, inspect coefficients
8. **Cross-Validation** — 3-fold CV for stable performance estimate
9. **Evaluation** — MAE, MSE, RMSE, R², Adjusted R²
10. **Residual Analysis** — KDE plot, residuals vs predicted scatter
11. **OLS Summary** — statsmodels for p-values, F-stat, confidence intervals
12. **New Data Prediction** — correct scaling before predicting

---

## 📊 Results

| Metric | Value |
|--------|-------|
| MAE | ~60 index points |
| RMSE | ~76 index points |
| R² | 0.83 |
| Adjusted R² | 0.71 |
| CV RMSE (3-fold) | ~77 index points |

### Key findings:
- The model explains **83% of variance** in index price
- **Unemployment rate** is the more significant predictor (p = 0.015 ✅)
- **Interest rate** is borderline significant (p = 0.054)
- CV RMSE ≈ Test RMSE → **model is stable, not overfitting** ✅
- RMSE > MAE → one outlier (+149 residual) inflating error

---

## ⚠️ Challenges & Fixes

### 1. Prediction mismatch (biggest bug)
**Problem:** Prediction gave `680` instead of the expected `~1464`.  
**Cause:** The model was trained on scaled data, but the new input was fed in raw (unscaled).  
**Fix:** Always apply `scaler.transform()` to new data before predicting. Never re-fit the scaler.

```python
# ❌ Wrong
new_data_ols = sm.add_constant(new_data_raw, has_constant='add')

# ✅ Correct
new_data_scaled = scaler.transform(new_data_raw)
new_data_ols = sm.add_constant(new_data_scaled, has_constant='add')
```

### 2. Data leakage in scaling
**Problem:** If you `fit_transform` on the entire dataset before splitting, test data information leaks into the scaler.  
**Fix:** Always `fit_transform` on `X_train` only, then only `transform` on `X_test`.

### 3. High multicollinearity
**Observation:** `interest_rate` and `unemployment_rate` have r = -0.93 with each other — both are strong predictors but also highly correlated, which can inflate standard errors.  
**Mitigation:** Noted in analysis; Ridge regression would be a better approach with more data.

---

## 🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/yourusername/multiple-linear-regression.git
cd multiple-linear-regression

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels jupyter

# Launch the notebook
jupyter notebook multiple_linear_regression_project.ipynb
```

---

## 🔮 Future Improvements

- [ ] Collect more data (24 rows is very small for ML)
- [ ] Add more economic features (inflation, GDP growth, etc.)
- [ ] Try Ridge/Lasso regression to handle multicollinearity
- [ ] Implement polynomial features for non-linear patterns
- [ ] Build an interactive prediction widget

---

## 🧠 What I Learned

- The difference between MAE, MSE, RMSE, and when each one matters
- Why scaling must happen **after** the train-test split to prevent data leakage
- How to interpret Adjusted R² vs regular R²
- Why cross-validation is more reliable than a single test split
- How to read a Statsmodels OLS summary (p-values, confidence intervals, F-statistic)
- How residual plots reveal what your model is missing

---

## 📬 Connect

If you have feedback or want to discuss the project, feel free to connect on [LinkedIn](#) or open an issue!

---

*Part of my Machine Learning from Scratch series. Previous project: [Simple Linear Regression](#)*
