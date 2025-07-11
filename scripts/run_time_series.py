#!/usr/bin/env python
"""
Run Time Series Analysis: Decomposition, ARIMA forecasting, plots
"""
import os
import sys
import glob
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA

def find_latest_data():
    data_files = glob.glob('data/processed/fred_data_*.csv')
    if not data_files:
        raise FileNotFoundError("No FRED data files found. Run the pipeline first.")
    return max(data_files, key=os.path.getctime)

def main():
    print("="*60)
    print("FRED Time Series Analysis: Decomposition & ARIMA Forecasting")
    print("="*60)
    data_file = find_latest_data()
    print(f"Using data file: {data_file}")
    df = pd.read_csv(data_file, index_col=0, parse_dates=True)
    target_var = 'GDP'
    if target_var not in df.columns:
        print(f"Target variable '{target_var}' not found in data.")
        return
    ts_data = df[target_var].dropna()
    if len(ts_data) < 50:
        print("Insufficient data for time series analysis (need at least 50 points). Skipping.")
        return
    print(f"Time series length: {len(ts_data)} observations")
    print(f"Date range: {ts_data.index.min()} to {ts_data.index.max()}")
    # Decomposition
    try:
        if ts_data.index.freq is None:
            ts_monthly = ts_data.resample('M').mean()
        else:
            ts_monthly = ts_data
        decomposition = seasonal_decompose(ts_monthly, model='additive', period=12)
        fig, axes = plt.subplots(4, 1, figsize=(12, 10))
        decomposition.observed.plot(ax=axes[0], title='Original Time Series')
        decomposition.trend.plot(ax=axes[1], title='Trend')
        decomposition.seasonal.plot(ax=axes[2], title='Seasonality')
        decomposition.resid.plot(ax=axes[3], title='Residuals')
        plt.tight_layout()
        plt.savefig('data/exports/time_series_decomposition.png', dpi=200, bbox_inches='tight')
        plt.close()
        print("Decomposition plot saved.")
    except Exception as e:
        print(f"Decomposition failed: {e}")
    # ARIMA Forecasting
    try:
        model = ARIMA(ts_monthly, order=(1, 1, 1))
        fitted_model = model.fit()
        print(f"ARIMA Model Summary:\n{fitted_model.summary()}")
        forecast_steps = min(12, len(ts_monthly) // 4)
        forecast = fitted_model.forecast(steps=forecast_steps)
        conf_int = fitted_model.get_forecast(steps=forecast_steps).conf_int()
        plt.figure(figsize=(12, 6))
        ts_monthly.plot(label='Historical Data')
        forecast.plot(label='Forecast', color='red')
        plt.fill_between(forecast.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], alpha=0.3, color='red', label='Confidence Interval')
        plt.title(f'{target_var} - ARIMA Forecast')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('data/exports/time_series_forecast.png', dpi=200, bbox_inches='tight')
        plt.close()
        print("Forecast plot saved.")
    except Exception as e:
        print(f"ARIMA modeling failed: {e}")
    print("\nTime series analysis complete. Outputs saved to data/exports/.")

if __name__ == "__main__":
    main() 