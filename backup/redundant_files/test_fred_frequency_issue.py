#!/usr/bin/env python3
"""
Test script to debug FRED API frequency parameter issue
"""

import os
import sys
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_enhanced_fred_client():
    """Test the enhanced FRED client to identify frequency parameter issue"""
    
    print("=== TESTING ENHANCED FRED CLIENT ===")
    
    # Get API key
    api_key = os.getenv('FRED_API_KEY')
    if not api_key:
        print("❌ FRED_API_KEY not set")
        return
    
    try:
        from src.core.enhanced_fred_client import EnhancedFREDClient
        
        # Initialize client
        client = EnhancedFREDClient(api_key)
        
        # Test problematic indicators
        problematic_indicators = ['GDPC1', 'INDPRO', 'RSAFS']
        
        print(f"\nTesting indicators: {problematic_indicators}")
        
        for indicator in problematic_indicators:
            print(f"\n--- Testing {indicator} ---")
            try:
                # Test direct series fetch
                series = client._fetch_series(
                    indicator, 
                    '2020-01-01', 
                    '2024-12-31', 
                    'auto'
                )
                
                if series is not None and not series.empty:
                    print(f"✅ {indicator}: Successfully fetched {len(series)} observations")
                    print(f"   Latest value: {series.iloc[-1]:.2f}")
                    print(f"   Date range: {series.index.min()} to {series.index.max()}")
                else:
                    print(f"❌ {indicator}: No data returned")
                    
            except Exception as e:
                print(f"❌ {indicator}: Error - {e}")
        
        # Test full data fetch
        print(f"\n--- Testing full data fetch ---")
        try:
            data = client.fetch_economic_data(
                indicators=problematic_indicators,
                start_date='2020-01-01',
                end_date='2024-12-31',
                frequency='auto'
            )
            
            print(f"✅ Full data fetch successful")
            print(f"   Shape: {data.shape}")
            print(f"   Columns: {list(data.columns)}")
            print(f"   Date range: {data.index.min()} to {data.index.max()}")
            
            # Show sample data
            print(f"\nSample data (last 3 observations):")
            print(data.tail(3))
            
        except Exception as e:
            print(f"❌ Full data fetch failed: {e}")
            
    except Exception as e:
        print(f"❌ Failed to import or initialize EnhancedFREDClient: {e}")

def test_fredapi_direct():
    """Test fredapi library directly"""
    
    print("\n=== TESTING FREDAPI LIBRARY DIRECTLY ===")
    
    try:
        from fredapi import Fred
        
        api_key = os.getenv('FRED_API_KEY')
        if not api_key:
            print("❌ FRED_API_KEY not set")
            return
        
        fred = Fred(api_key=api_key)
        
        # Test problematic indicators
        problematic_indicators = ['GDPC1', 'INDPRO', 'RSAFS']
        
        for indicator in problematic_indicators:
            print(f"\n--- Testing {indicator} with fredapi ---")
            try:
                # Test without any frequency parameter
                series = fred.get_series(
                    indicator,
                    observation_start='2020-01-01',
                    observation_end='2024-12-31'
                )
                
                if not series.empty:
                    print(f"✅ {indicator}: Successfully fetched {len(series)} observations")
                    print(f"   Latest value: {series.iloc[-1]:.2f}")
                    print(f"   Date range: {series.index.min()} to {series.index.max()}")
                else:
                    print(f"❌ {indicator}: No data returned")
                    
            except Exception as e:
                print(f"❌ {indicator}: Error - {e}")
                
    except Exception as e:
        print(f"❌ Failed to test fredapi directly: {e}")

if __name__ == "__main__":
    test_enhanced_fred_client()
    test_fredapi_direct() 