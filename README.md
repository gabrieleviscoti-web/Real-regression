# Real Regression – MPG Dataset
This project applies multiple linear regression to the mpg dataset to model fuel efficiency based on vehicle characteristics.
The analysis demonstrates skills in data cleaning, feature selection, model training, evaluation metrics, residual analysis, and multicollinearity diagnostics.

## Project Objective <br>
Build and evaluate a multiple linear regression model to predict mpg using the variables:

- horsepower

- weight

- acceleration

The project also includes residual analysis and multicollinearity checks using Variance Inflation Factor (VIF).

# Dataset
The dataset is loaded from the seaborn library:

```python
import pandas as pd
import seaborn as sns

df = sns.load_dataset("mpg")
```
Missing values are removed:

```python
df = df.dropna()
```
## Feature Selection
```python
X = df[['horsepower', 'weight', 'acceleration']]
y = df['mpg']
```
## Train/Test Split
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```
## Regression Model
```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
```
Model parameters include:

- coefficients

- intercept

- number of features

- rank

- singular values

## Prediction
```python
y_pred = model.predict(X_test)
```
## Evaluation Metrics
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

rmse = mean_squared_error(y_test, y_pred) ** 0.5
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(rmse, mae, r2)
```
Typical output:

- RMSE ≈ 4.22

- MAE ≈ 3.50

- R² ≈ 0.65

These values indicate a moderately good fit for a simple linear model.

## Residual Analysis
```python
import matplotlib.pyplot as plt
import seaborn as sns

residuals = y_test - y_pred
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(0, color='red')
plt.title("Residuals Plot")
plt.show()
```
Residual analysis helps verify:

- linearity

- homoscedasticity

- absence of strong patterns

## Multicollinearity (VIF)
```python
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

X_vif = sm.add_constant(X)
vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
vif["feature"] = X_vif.columns
print(vif)
```
Typical VIF results:

- horsepower → ~8.2

- weight → ~5.2

- acceleration → ~2.5

Values above 5 indicate potential multicollinearity.

## Key Results
- The regression model explains ~65% of variance in mpg.

- Horsepower and weight have the strongest negative impact on mpg.

- Residuals show acceptable dispersion around zero.

- VIF reveals moderate multicollinearity between predictors.

The model is suitable for exploratory analysis and baseline prediction.

## Technologies Used
- Python

- Pandas

- Seaborn

- Matplotlib

- Scikit‑learn

- Statsmodels

## Purpose of This Project
This project demonstrates:

- ability to build and evaluate regression models

- understanding of model diagnostics

- correct use of residual plots

- multicollinearity analysis

- reproducible machine learning workflow

## Possible Extensions
- Polynomial regression

- Regularization (Ridge, Lasso)

- Feature engineering

- Cross‑validation

- Comparison with tree‑based models
