#!/usr/bin/env python
"""
Run Statistical Modeling: Linear regression, diagnostics, p-values, confidence intervals, plots
"""
import os
import sys
import glob
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

def find_latest_data():
    data_files = glob.glob('data/processed/fred_data_*.csv')
    if not data_files:
        raise FileNotFoundError("No FRED data files found. Run the pipeline first.")
    return max(data_files, key=os.path.getctime)

def main():
    print("="*60)
    print("FRED Statistical Modeling: Linear Regression & Diagnostics")
    print("="*60)
    data_file = find_latest_data()
    print(f"Using data file: {data_file}")
    df = pd.read_csv(data_file, index_col=0, parse_dates=True)
    df_clean = df.dropna()
    target_var = 'GDP'
    if target_var not in df_clean.columns:
        print(f"Target variable '{target_var}' not found in data.")
        return
    feature_cols = [col for col in df_clean.columns if col != target_var]
    X = df_clean[feature_cols]
    y = df_clean[target_var]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Fit linear regression
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    # Model performance
    r2_train = model.score(X_train, y_train)
    r2_test = model.score(X_test, y_test)
    print(f"R² (Train): {r2_train:.4f} | R² (Test): {r2_test:.4f}")
    # Coefficients
    print("\nCoefficients:")
    for feature, coef in zip(feature_cols, model.coef_):
        print(f"  {feature}: {coef:.4f}")
    print(f"  Intercept: {model.intercept_:.4f}")
    # Statsmodels for p-values and CIs
    X_with_const = sm.add_constant(X_train)
    model_sm = sm.OLS(y_train, X_with_const).fit()
    print("\nStatistical Significance:")
    print(model_sm.summary().tables[1])
    # Save summary table
    with open('data/exports/regression_summary.txt', 'w') as f:
        f.write(str(model_sm.summary()))
    # Residuals
    residuals = y_train - y_pred_train
    # Normality test
    _, p_value_norm = stats.normaltest(residuals)
    print(f"Normality test (p-value): {p_value_norm:.4f}")
    # VIF
    vif_data = []
    for i in range(X_train.shape[1]):
        try:
            vif = variance_inflation_factor(X_train.values, i)
            vif_data.append(vif)
        except:
            vif_data.append(np.nan)
    print("\nVariance Inflation Factors:")
    for feature, vif in zip(feature_cols, vif_data):
        print(f"  {feature}: {vif:.3f}")
    # Homoscedasticity
    try:
        _, p_value_het = het_breuschpagan(residuals, X_with_const)
        print(f"Homoscedasticity test (p-value): {p_value_het:.4f}")
    except:
        print("Homoscedasticity test failed")
    # Diagnostic plots
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.scatter(y_pred_train, residuals, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Fitted')
    plt.subplot(1,2,2)
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Normal Q-Q')
    plt.tight_layout()
    plt.savefig('data/exports/regression_diagnostics.png', dpi=200)
    plt.close()
    print("\nStatistical modeling complete. Outputs saved to data/exports/.")

if __name__ == "__main__":
    main() 