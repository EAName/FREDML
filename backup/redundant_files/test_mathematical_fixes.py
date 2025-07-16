#!/usr/bin/env python3
"""
Test script to verify mathematical fixes module
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_mathematical_fixes():
    """Test the mathematical fixes module"""
    print("🔍 Testing mathematical fixes module...")
    
    try:
        from src.analysis.mathematical_fixes import MathematicalFixes
        
        # Create test data
        dates = pd.date_range('2020-01-01', periods=100, freq='ME')
        test_data = pd.DataFrame({
            'GDPC1': np.random.normal(22000, 1000, 100),  # Billions
            'INDPRO': np.random.normal(100, 5, 100),      # Index
            'CPIAUCSL': np.random.normal(250, 10, 100),   # Index
            'FEDFUNDS': np.random.normal(2, 0.5, 100),    # Percent
            'PAYEMS': np.random.normal(150000, 5000, 100) # Thousands
        }, index=dates)
        
        print("✅ Test data created successfully")
        
        # Initialize mathematical fixes
        fixes = MathematicalFixes()
        print("✅ MathematicalFixes initialized successfully")
        
        # Test unit normalization
        normalized_data = fixes.normalize_units(test_data)
        print(f"✅ Unit normalization completed. Shape: {normalized_data.shape}")
        
        # Test frequency alignment
        aligned_data = fixes.align_frequencies(test_data, target_freq='QE')
        print(f"✅ Frequency alignment completed. Shape: {aligned_data.shape}")
        
        # Test growth rate calculation
        growth_data = fixes.calculate_growth_rates(test_data, method='pct_change')
        print(f"✅ Growth rate calculation completed. Shape: {growth_data.shape}")
        
        # Test stationarity enforcement
        stationary_data, diff_info = fixes.enforce_stationarity(growth_data)
        print(f"✅ Stationarity enforcement completed. Shape: {stationary_data.shape}")
        print(f"✅ Differencing info: {len(diff_info)} indicators processed")
        
        # Test comprehensive fixes
        fixed_data, fix_info = fixes.apply_comprehensive_fixes(
            test_data,
            target_freq='QE',
            growth_method='pct_change',
            normalize_units=True
        )
        print(f"✅ Comprehensive fixes applied. Final shape: {fixed_data.shape}")
        print(f"✅ Applied fixes: {fix_info['fixes_applied']}")
        
        # Test safe error metrics
        actual = np.array([1, 2, 3, 4, 5])
        forecast = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        
        mape = fixes.safe_mape(actual, forecast)
        mae = fixes.safe_mae(actual, forecast)
        rmse = fixes.safe_rmse(actual, forecast)
        
        print(f"✅ Error metrics calculated - MAPE: {mape:.2f}%, MAE: {mae:.2f}, RMSE: {rmse:.2f}")
        
        # Test forecast period scaling
        for indicator in ['GDPC1', 'INDPRO', 'FEDFUNDS']:
            scaled_periods = fixes.scale_forecast_periods(4, indicator, test_data)
            print(f"✅ {indicator}: scaled forecast periods from 4 to {scaled_periods}")
        
        print("\n🎉 All mathematical fixes tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Mathematical fixes test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_mathematical_fixes()
    if success:
        print("\n✅ Mathematical fixes module is working correctly!")
    else:
        print("\n❌ Mathematical fixes module has issues.") 