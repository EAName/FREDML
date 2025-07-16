#!/usr/bin/env python3
"""
Test script to verify GDP scale and fix the issue
"""

import os
import sys
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_gdp_scale():
    """Test GDP scale to ensure it matches FRED values"""
    
    print("=== TESTING GDP SCALE ===")
    
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
        
        # Fetch raw GDP data
        print("\n1. Fetching raw GDP data from FRED...")
        raw_data = client.fetch_economic_data(['GDPC1'], '2024-01-01', '2025-12-31')
        
        if raw_data.empty:
            print("❌ No raw data available")
            return
        
        print(f"Raw GDP data shape: {raw_data.shape}")
        print(f"Raw GDP values: {raw_data['GDPC1'].tail()}")
        
        # Apply mathematical fixes
        print("\n2. Applying mathematical fixes...")
        fixed_data, fix_info = math_fixes.apply_comprehensive_fixes(
            raw_data,
            target_freq='Q',
            growth_method='pct_change',
            normalize_units=True,
            preserve_absolute_values=True
        )
        
        print(f"Fixed data shape: {fixed_data.shape}")
        print(f"Fixed GDP values: {fixed_data['GDPC1'].tail()}")
        
        # Check if the values are in the correct range (should be ~23,500 billion)
        latest_gdp = fixed_data['GDPC1'].iloc[-1]
        print(f"\nLatest GDP value: {latest_gdp}")
        
        if 20000 <= latest_gdp <= 25000:
            print("✅ GDP scale is correct (in billions)")
        elif 20 <= latest_gdp <= 25:
            print("❌ GDP scale is wrong - showing in trillions instead of billions")
            print("   Expected: ~23,500 billion, Got: ~23.5 billion")
        else:
            print(f"❌ GDP scale is wrong - unexpected value: {latest_gdp}")
        
        # Test the unit normalization specifically
        print("\n3. Testing unit normalization...")
        normalized_data = math_fixes.normalize_units(raw_data)
        print(f"Normalized GDP values: {normalized_data['GDPC1'].tail()}")
        
        # Check the unit factors
        print(f"\n4. Current unit factors:")
        for indicator, factor in math_fixes.unit_factors.items():
            print(f"   {indicator}: {factor}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_gdp_scale() 