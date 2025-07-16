#!/usr/bin/env python3
"""
Test Mathematical Fixes - Fixed Version
Verify that the corrected unit normalization factors produce accurate data values
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import pandas as pd
import numpy as np
from src.analysis.mathematical_fixes import MathematicalFixes

def test_mathematical_fixes():
    """Test that mathematical fixes produce correct data values"""
    print("🧪 Testing Mathematical Fixes - Fixed Version")
    print("=" * 60)
    
    # Create sample data that matches FRED's actual values
    dates = pd.date_range('2024-01-01', periods=12, freq='M')
    
    # Sample data with realistic FRED values
    sample_data = pd.DataFrame({
        'GDPC1': [23500, 23550, 23600, 23650, 23700, 23750, 23800, 23850, 23900, 23950, 24000, 24050],  # Billions
        'CPIAUCSL': [310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321],  # Index ~320
        'INDPRO': [110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121],  # Index ~110-115
        'FEDFUNDS': [4.25, 4.30, 4.35, 4.40, 4.45, 4.50, 4.55, 4.60, 4.65, 4.70, 4.75, 4.80],  # Percent ~4.33%
        'DGS10': [3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9],  # Percent ~4.0%
        'RSAFS': [700000, 710000, 720000, 730000, 740000, 750000, 760000, 770000, 780000, 790000, 800000, 810000]  # Millions
    }, index=dates)
    
    print("📊 Original Data (Realistic FRED Values):")
    print(sample_data.head())
    print()
    
    # Initialize mathematical fixes
    math_fixes = MathematicalFixes()
    
    # Apply comprehensive fixes
    print("🔧 Applying Mathematical Fixes...")
    fixed_data, fix_info = math_fixes.apply_comprehensive_fixes(
        sample_data,
        target_freq='Q',
        growth_method='pct_change',
        normalize_units=True
    )
    
    print("✅ Fixes Applied:")
    for fix in fix_info['fixes_applied']:
        print(f"  - {fix}")
    print()
    
    # Test unit normalization specifically
    print("🧮 Testing Unit Normalization:")
    normalized_data = math_fixes.normalize_units(sample_data)
    
    print("Original vs Normalized Values:")
    for col in ['GDPC1', 'CPIAUCSL', 'INDPRO', 'FEDFUNDS', 'DGS10', 'RSAFS']:
        if col in sample_data.columns:
            original_val = sample_data[col].iloc[-1]
            normalized_val = normalized_data[col].iloc[-1]
            print(f"  {col}: {original_val:,.2f} → {normalized_val:,.2f}")
    
    print()
    
    # Verify the values are now correct
    print("✅ Expected vs Actual Values:")
    expected_values = {
        'GDPC1': (23500, 24050),  # Should be ~$23.5T (in billions)
        'CPIAUCSL': (310, 321),   # Should be ~320
        'INDPRO': (110, 121),     # Should be ~110-115
        'FEDFUNDS': (4.25, 4.80), # Should be ~4.33%
        'DGS10': (3.8, 4.9),     # Should be ~4.0%
        'RSAFS': (700, 810)       # Should be ~$700-900B (in billions)
    }
    
    for col, (min_expected, max_expected) in expected_values.items():
        if col in normalized_data.columns:
            actual_val = normalized_data[col].iloc[-1]
            if min_expected <= actual_val <= max_expected:
                print(f"  ✅ {col}: {actual_val:,.2f} (within expected range {min_expected:,.2f}-{max_expected:,.2f})")
            else:
                print(f"  ❌ {col}: {actual_val:,.2f} (outside expected range {min_expected:,.2f}-{max_expected:,.2f})")
    
    print()
    print("🎯 Mathematical Fixes Test Complete!")
    
    return fixed_data, fix_info

if __name__ == "__main__":
    test_mathematical_fixes() 