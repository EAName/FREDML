#!/usr/bin/env python3
"""
Test script to verify data accuracy against FRED values
"""

import os
import sys
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_data_accuracy():
    """Test data accuracy against known FRED values"""
    
    print("=== TESTING DATA ACCURACY ===")
    
    # Get API key
    api_key = os.getenv('FRED_API_KEY')
    if not api_key:
        print("❌ FRED_API_KEY not set")
        return
    
    try:
        from src.core.enhanced_fred_client import EnhancedFREDClient
        from src.analysis.mathematical_fixes import MathematicalFixes
        
        # Initialize client and mathematical fixes
        client = EnhancedFREDClient(api_key)
        math_fixes = MathematicalFixes()
        
        # Test indicators with known values
        test_indicators = ['GDPC1', 'CPIAUCSL', 'UNRATE']
        
        print(f"\nTesting indicators: {test_indicators}")
        
        # Fetch raw data
        raw_data = client.fetch_economic_data(
            indicators=test_indicators,
            start_date='2024-01-01',
            end_date='2024-12-31',
            frequency='auto'
        )
        
        print(f"\nRaw data shape: {raw_data.shape}")
        print(f"Raw data columns: {list(raw_data.columns)}")
        
        if not raw_data.empty:
            print(f"\nLatest raw values:")
            for indicator in test_indicators:
                if indicator in raw_data.columns:
                    latest_value = raw_data[indicator].dropna().iloc[-1]
                    print(f"  {indicator}: {latest_value:.2f}")
        
        # Apply mathematical fixes
        fixed_data, fix_info = math_fixes.apply_comprehensive_fixes(
            raw_data,
            target_freq='Q',
            growth_method='pct_change',
            normalize_units=True
        )
        
        print(f"\nFixed data shape: {fixed_data.shape}")
        print(f"Applied fixes: {fix_info}")
        
        if not fixed_data.empty:
            print(f"\nLatest fixed values:")
            for indicator in test_indicators:
                if indicator in fixed_data.columns:
                    latest_value = fixed_data[indicator].dropna().iloc[-1]
                    print(f"  {indicator}: {latest_value:.2f}")
        
        # Expected values based on your feedback
        expected_values = {
            'GDPC1': 23500,  # Should be ~23.5 trillion
            'CPIAUCSL': 316,  # Should be ~316
            'UNRATE': 3.7     # Should be ~3.7%
        }
        
        print(f"\nExpected values (from your feedback):")
        for indicator, expected in expected_values.items():
            print(f"  {indicator}: {expected}")
        
        # Compare with actual values
        print(f"\nAccuracy check:")
        for indicator in test_indicators:
            if indicator in fixed_data.columns:
                actual_value = fixed_data[indicator].dropna().iloc[-1]
                expected_value = expected_values.get(indicator, 0)
                
                if expected_value > 0:
                    accuracy = abs(actual_value - expected_value) / expected_value * 100
                    print(f"  {indicator}: {actual_value:.2f} vs {expected_value:.2f} (accuracy: {accuracy:.1f}%)")
                else:
                    print(f"  {indicator}: {actual_value:.2f} (no expected value)")
        
        # Test unit normalization factors
        print(f"\nUnit normalization factors:")
        for indicator in test_indicators:
            factor = math_fixes.unit_factors.get(indicator, 1)
            print(f"  {indicator}: factor = {factor}")
        
    except Exception as e:
        print(f"❌ Failed to test data accuracy: {e}")

if __name__ == "__main__":
    test_data_accuracy() 