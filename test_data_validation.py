#!/usr/bin/env python3
"""
Data Validation Script
Test the economic indicators and identify math issues
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.core.enhanced_fred_client import EnhancedFREDClient

def test_data_validation():
    """Test data validation and identify issues"""
    
    # Use a demo API key for testing (FRED allows limited access without key)
    api_key = "demo"  # FRED demo key for testing
    
    print("=== ECONOMIC DATA VALIDATION TEST ===\n")
    
    try:
        # Initialize client
        client = EnhancedFREDClient(api_key)
        
        # Test indicators
        indicators = ['GDPC1', 'CPIAUCSL', 'INDPRO', 'RSAFS', 'FEDFUNDS', 'DGS10']
        
        print("1. Testing data fetching...")
        data = client.fetch_economic_data(
            indicators=indicators,
            start_date='2020-01-01',
            end_date='2024-12-31',
            frequency='auto'
        )
        
        print(f"Data shape: {data.shape}")
        print(f"Date range: {data.index.min()} to {data.index.max()}")
        print(f"Columns: {list(data.columns)}")
        
        print("\n2. Raw data sample (last 5 observations):")
        print(data.tail())
        
        print("\n3. Data statistics:")
        print(data.describe())
        
        print("\n4. Missing data analysis:")
        missing_data = data.isnull().sum()
        print(missing_data)
        
        print("\n5. Testing frequency standardization...")
        # Test the frequency standardization
        for indicator in indicators:
            if indicator in data.columns:
                series = data[indicator].dropna()
                print(f"{indicator}: {len(series)} observations, freq: {series.index.freq}")
        
        print("\n6. Testing growth rate calculation...")
        # Test growth rate calculation
        for indicator in indicators:
            if indicator in data.columns:
                series = data[indicator].dropna()
                if len(series) > 1:
                    # Calculate percent change
                    pct_change = series.pct_change().dropna()
                    latest_change = pct_change.iloc[-1] * 100 if len(pct_change) > 0 else 0
                    print(f"{indicator}: Latest change = {latest_change:.2f}%")
                    print(f"  Raw values: {series.iloc[-2]:.2f} -> {series.iloc[-1]:.2f}")
        
        print("\n7. Testing unit normalization...")
        # Test unit normalization
        for indicator in indicators:
            if indicator in data.columns:
                series = data[indicator].dropna()
                if len(series) > 0:
                    mean_val = series.mean()
                    std_val = series.std()
                    print(f"{indicator}: Mean={mean_val:.2f}, Std={std_val:.2f}")
                    
                    # Check for potential unit issues
                    if mean_val > 1000000:  # Likely in billions/trillions
                        print(f"  WARNING: {indicator} has very large values - may need unit conversion")
                    elif mean_val < 1 and indicator in ['FEDFUNDS', 'DGS10']:
                        print(f"  WARNING: {indicator} has small values - may be in decimal form instead of percentage")
        
        print("\n8. Testing data quality validation...")
        quality_report = client.validate_data_quality(data)
        print("Quality report summary:")
        for series, metrics in quality_report['missing_data'].items():
            print(f"  {series}: {metrics['completeness']:.1f}% complete")
        
        print("\n9. Testing frequency alignment...")
        # Check if all series have the same frequency
        frequencies = {}
        for indicator in indicators:
            if indicator in data.columns:
                series = data[indicator].dropna()
                if len(series) > 0:
                    freq = pd.infer_freq(series.index)
                    frequencies[indicator] = freq
                    print(f"  {indicator}: {freq}")
        
        # Check for frequency mismatches
        unique_freqs = set(frequencies.values())
        if len(unique_freqs) > 1:
            print(f"  WARNING: Multiple frequencies detected: {unique_freqs}")
            print("  This may cause issues in modeling and forecasting")
        
        print("\n=== VALIDATION COMPLETE ===")
        
        # Summary of potential issues
        print("\n=== POTENTIAL ISSUES IDENTIFIED ===")
        
        issues = []
        
        # Check for unit scale issues
        for indicator in indicators:
            if indicator in data.columns:
                series = data[indicator].dropna()
                if len(series) > 0:
                    mean_val = series.mean()
                    if mean_val > 1000000:
                        issues.append(f"Unit scale issue: {indicator} has very large values ({mean_val:.0f})")
                    elif mean_val < 1 and indicator in ['FEDFUNDS', 'DGS10']:
                        issues.append(f"Unit format issue: {indicator} may be in decimal form instead of percentage")
        
        # Check for frequency issues
        if len(unique_freqs) > 1:
            issues.append(f"Frequency mismatch: Series have different frequencies {unique_freqs}")
        
        # Check for missing data
        for series, metrics in quality_report['missing_data'].items():
            if metrics['missing_percentage'] > 10:
                issues.append(f"Missing data: {series} has {metrics['missing_percentage']:.1f}% missing values")
        
        if issues:
            for issue in issues:
                print(f"  • {issue}")
        else:
            print("  No major issues detected")
            
    except Exception as e:
        print(f"Error during validation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_data_validation() 