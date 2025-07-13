#!/usr/bin/env python3
"""
Math Issues Demonstration
Demonstrate the specific math problems identified in the economic analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_mock_economic_data():
    """Create mock economic data to demonstrate the issues"""
    
    # Create date range
    dates = pd.date_range('2020-01-01', '2024-12-31', freq='Q')
    
    # Mock data representing the actual issues
    data = {
        'GDPC1': [22000, 22100, 22200, 22300, 22400, 22500, 22600, 22700, 22800, 22900, 23000, 23100, 23200, 23300, 23400, 23500, 23600, 23700, 23800, 23900],  # Billions
        'CPIAUCSL': [258.0, 258.5, 259.0, 259.5, 260.0, 260.5, 261.0, 261.5, 262.0, 262.5, 263.0, 263.5, 264.0, 264.5, 265.0, 265.5, 266.0, 266.5, 267.0, 267.5],  # Index
        'INDPRO': [100.0, 100.5, 101.0, 101.5, 102.0, 102.5, 103.0, 103.5, 104.0, 104.5, 105.0, 105.5, 106.0, 106.5, 107.0, 107.5, 108.0, 108.5, 109.0, 109.5],  # Index
        'RSAFS': [500000, 502000, 504000, 506000, 508000, 510000, 512000, 514000, 516000, 518000, 520000, 522000, 524000, 526000, 528000, 530000, 532000, 534000, 536000, 538000],  # Millions
        'FEDFUNDS': [0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27],  # Decimal form
        'DGS10': [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4]  # Decimal form
    }
    
    df = pd.DataFrame(data, index=dates)
    return df

def demonstrate_issues():
    """Demonstrate the specific math issues"""
    
    print("=== ECONOMIC INDICATORS MATH ISSUES DEMONSTRATION ===\n")
    
    # Create mock data
    data = create_mock_economic_data()
    
    print("1. RAW DATA (showing the issues):")
    print(data.tail())
    print()
    
    print("2. DATA STATISTICS (revealing scale problems):")
    print(data.describe())
    print()
    
    # Issue 1: Unit Scale Problems
    print("3. UNIT SCALE ISSUES:")
    print("   • GDPC1: Values in billions (22,000 = $22 trillion)")
    print("   • RSAFS: Values in millions (500,000 = $500 billion)")
    print("   • CPIAUCSL: Index values (~260)")
    print("   • FEDFUNDS: Decimal form (0.08 = 8%)")
    print("   • DGS10: Decimal form (1.5 = 1.5%)")
    print()
    
    # Issue 2: Growth Rate Calculation Problems
    print("4. GROWTH RATE CALCULATION ISSUES:")
    for col in data.columns:
        series = data[col]
        # Calculate both absolute change and percent change
        abs_change = series.iloc[-1] - series.iloc[-2]
        pct_change = ((series.iloc[-1] - series.iloc[-2]) / series.iloc[-2]) * 100
        
        print(f"   {col}:")
        print(f"     Raw values: {series.iloc[-2]:.2f} -> {series.iloc[-1]:.2f}")
        print(f"     Absolute change: {abs_change:.2f}")
        print(f"     Percent change: {pct_change:.2f}%")
        
        # Show the problem with interpretation
        if col == 'GDPC1':
            print(f"     PROBLEM: This shows as +100 (absolute) but should be +0.45% (relative)")
        elif col == 'FEDFUNDS':
            print(f"     PROBLEM: This shows as +0.01 (absolute) but should be +11.11% (relative)")
        print()
    
    # Issue 3: Frequency Problems
    print("5. FREQUENCY ALIGNMENT ISSUES:")
    print("   • GDPC1: Quarterly data")
    print("   • CPIAUCSL: Monthly data (resampled to quarterly)")
    print("   • INDPRO: Monthly data (resampled to quarterly)")
    print("   • RSAFS: Monthly data (resampled to quarterly)")
    print("   • FEDFUNDS: Daily data (resampled to quarterly)")
    print("   • DGS10: Daily data (resampled to quarterly)")
    print("   PROBLEM: Different original frequencies may cause misalignment")
    print()
    
    # Issue 4: Missing Normalization
    print("6. MISSING UNIT NORMALIZATION:")
    print("   Without normalization, large-scale variables dominate:")
    
    # Calculate correlations without normalization
    growth_data = data.pct_change().dropna()
    corr_matrix = growth_data.corr()
    
    print("   Correlation matrix (without normalization):")
    print(corr_matrix.round(3))
    print()
    
    # Show how normalization would help
    print("7. NORMALIZED DATA (how it should look):")
    normalized_data = (data - data.mean()) / data.std()
    print(normalized_data.tail())
    print()
    
    # Issue 5: MAPE Calculation Problems
    print("8. MAPE CALCULATION ISSUES:")
    
    # Simulate forecasting results
    actual = np.array([100, 101, 102, 103, 104])
    forecast = np.array([99, 100.5, 101.8, 102.9, 103.8])
    
    # Calculate MAPE
    mape = np.mean(np.abs((actual - forecast) / actual)) * 100
    
    print(f"   Actual values: {actual}")
    print(f"   Forecast values: {forecast}")
    print(f"   MAPE: {mape:.2f}%")
    
    # Show the problem with zero or near-zero values
    actual_with_zero = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    forecast_with_zero = np.array([0.15, 0.25, 0.35, 0.45, 0.55])
    
    try:
        mape_with_zero = np.mean(np.abs((actual_with_zero - forecast_with_zero) / actual_with_zero)) * 100
        print(f"   MAPE with small values: {mape_with_zero:.2f}% (can be unstable)")
    except:
        print("   MAPE with small values: ERROR (division by zero)")
    
    print()
    
    # Issue 6: Forecast Period Problems
    print("9. FORECAST PERIOD ISSUES:")
    print("   • Default forecast_periods=4")
    print("   • For quarterly data: 4 quarters = 1 year (reasonable)")
    print("   • For daily data: 4 days = 4 days (too short)")
    print("   • For monthly data: 4 months = 4 months (reasonable)")
    print("   PROBLEM: Same horizon applied to different frequencies")
    print()
    
    # Issue 7: Stationarity Problems
    print("10. STATIONARITY ISSUES:")
    print("   • Raw economic data is typically non-stationary")
    print("   • GDP, CPI, Industrial Production all have trends")
    print("   • Granger causality tests require stationarity")
    print("   • PROBLEM: Tests run on raw data instead of differenced data")
    print()
    
    # Summary of fixes needed
    print("=== RECOMMENDED FIXES ===")
    print("1. Unit Normalization:")
    print("   • Apply z-score normalization: (x - mean) / std")
    print("   • Or use log transformations for growth rates")
    print()
    
    print("2. Frequency Alignment:")
    print("   • Resample all series to common frequency (e.g., quarterly)")
    print("   • Use appropriate aggregation methods (mean for rates, last for levels)")
    print()
    
    print("3. Growth Rate Calculation:")
    print("   • Explicitly calculate percent changes: series.pct_change() * 100")
    print("   • Ensure proper interpretation of results")
    print()
    
    print("4. Forecast Period Scaling:")
    print("   • Scale forecast periods by frequency:")
    print("   • Daily: periods * 90 (for quarterly equivalent)")
    print("   • Monthly: periods * 3 (for quarterly equivalent)")
    print("   • Quarterly: periods * 1 (no change)")
    print()
    
    print("5. Safe MAPE Calculation:")
    print("   • Add small epsilon to denominator: np.maximum(np.abs(actual), 1e-5)")
    print("   • Include MAE and RMSE alongside MAPE")
    print()
    
    print("6. Stationarity Enforcement:")
    print("   • Test for stationarity using ADF test")
    print("   • Difference non-stationary series before Granger tests")
    print("   • Use SARIMA for seasonal series")
    print()

if __name__ == "__main__":
    demonstrate_issues() 