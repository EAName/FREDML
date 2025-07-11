#!/usr/bin/env python3
"""
Quick Start Guide for FRED Economic Data Analysis
Demonstrates how to load and analyze the collected data
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.fred_client import FREDDataCollectorV2
from datetime import datetime, timedelta

def load_latest_data():
    """Load the most recent data file."""
    import os
    import glob
    
    # Find the most recent data file
    data_files = glob.glob('data/fred_economic_data_*.csv')
    if not data_files:
        print("No data files found. Run the collector first.")
        return None
    
    latest_file = max(data_files, key=os.path.getctime)
    print(f"Loading data from: {latest_file}")
    
    df = pd.read_csv(latest_file, index_col=0, parse_dates=True)
    return df

def analyze_gdp_trends(df):
    """Analyze GDP trends."""
    print("\n=== GDP Analysis ===")
    
    if 'GDP' not in df.columns:
        print("GDP data not available")
        return
    
    gdp_data = df['GDP'].dropna()
    
    print(f"GDP Data Points: {len(gdp_data)}")
    print(f"Date Range: {gdp_data.index.min()} to {gdp_data.index.max()}")
    print(f"Latest GDP: ${gdp_data.iloc[-1]:,.2f} billion")
    print(f"GDP Growth (last 5 years): {((gdp_data.iloc[-1] / gdp_data.iloc[-20]) - 1) * 100:.2f}%")
    
    # Plot GDP trend
    plt.figure(figsize=(12, 6))
    gdp_data.plot(linewidth=2)
    plt.title('US GDP Over Time')
    plt.ylabel('GDP (Billions of Dollars)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def analyze_unemployment(df):
    """Analyze unemployment trends."""
    print("\n=== Unemployment Analysis ===")
    
    if 'UNRATE' not in df.columns:
        print("Unemployment data not available")
        return
    
    unrate_data = df['UNRATE'].dropna()
    
    print(f"Unemployment Data Points: {len(unrate_data)}")
    print(f"Current Unemployment Rate: {unrate_data.iloc[-1]:.1f}%")
    print(f"Average Unemployment Rate: {unrate_data.mean():.1f}%")
    print(f"Lowest Rate: {unrate_data.min():.1f}%")
    print(f"Highest Rate: {unrate_data.max():.1f}%")
    
    # Plot unemployment trend
    plt.figure(figsize=(12, 6))
    unrate_data.plot(linewidth=2, color='red')
    plt.title('US Unemployment Rate Over Time')
    plt.ylabel('Unemployment Rate (%)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def analyze_inflation(df):
    """Analyze inflation trends using CPI."""
    print("\n=== Inflation Analysis (CPI) ===")
    
    if 'CPIAUCSL' not in df.columns:
        print("CPI data not available")
        return
    
    cpi_data = df['CPIAUCSL'].dropna()
    
    # Calculate year-over-year inflation
    cpi_yoy = cpi_data.pct_change(periods=12) * 100
    
    print(f"CPI Data Points: {len(cpi_data)}")
    print(f"Current CPI: {cpi_data.iloc[-1]:.2f}")
    print(f"Current YoY Inflation: {cpi_yoy.iloc[-1]:.2f}%")
    print(f"Average YoY Inflation: {cpi_yoy.mean():.2f}%")
    
    # Plot inflation trend
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    cpi_data.plot(ax=ax1, linewidth=2, color='green')
    ax1.set_title('Consumer Price Index (CPI)')
    ax1.set_ylabel('CPI')
    ax1.grid(True, alpha=0.3)
    
    cpi_yoy.plot(ax=ax2, linewidth=2, color='orange')
    ax2.set_title('Year-over-Year Inflation Rate')
    ax2.set_ylabel('Inflation Rate (%)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def analyze_interest_rates(df):
    """Analyze interest rate trends."""
    print("\n=== Interest Rate Analysis ===")
    
    rates_data = {}
    if 'FEDFUNDS' in df.columns:
        rates_data['Federal Funds Rate'] = df['FEDFUNDS'].dropna()
    if 'DGS10' in df.columns:
        rates_data['10-Year Treasury'] = df['DGS10'].dropna()
    
    if not rates_data:
        print("No interest rate data available")
        return
    
    for name, data in rates_data.items():
        print(f"\n{name}:")
        print(f"  Current Rate: {data.iloc[-1]:.2f}%")
        print(f"  Average Rate: {data.mean():.2f}%")
        print(f"  Range: {data.min():.2f}% - {data.max():.2f}%")
    
    # Plot interest rates
    plt.figure(figsize=(12, 6))
    for name, data in rates_data.items():
        data.plot(linewidth=2, label=name)
    
    plt.title('Interest Rates Over Time')
    plt.ylabel('Interest Rate (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def correlation_analysis(df):
    """Analyze correlations between economic indicators."""
    print("\n=== Correlation Analysis ===")
    
    # Select available indicators
    available_cols = [col for col in ['GDP', 'UNRATE', 'CPIAUCSL', 'FEDFUNDS', 'DGS10'] 
                     if col in df.columns]
    
    if len(available_cols) < 2:
        print("Need at least 2 indicators for correlation analysis")
        return
    
    # Calculate correlations
    corr_data = df[available_cols].corr()
    
    print("Correlation Matrix:")
    print(corr_data.round(3))
    
    # Plot correlation heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5)
    plt.title('Economic Indicators Correlation Matrix')
    plt.tight_layout()
    plt.show()

def main():
    """Run the quick start analysis."""
    print("FRED Economic Data - Quick Start Analysis")
    print("=" * 50)
    
    # Load data
    df = load_latest_data()
    if df is None:
        return
    
    print(f"Data loaded successfully!")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    # Run analyses
    analyze_gdp_trends(df)
    analyze_unemployment(df)
    analyze_inflation(df)
    analyze_interest_rates(df)
    correlation_analysis(df)
    
    print("\n=== Analysis Complete ===")
    print("Check the generated plots for visual insights!")

if __name__ == "__main__":
    main() 