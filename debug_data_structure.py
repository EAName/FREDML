#!/usr/bin/env python3
"""
Debug script to check the actual data structure and values
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.core.enhanced_fred_client import EnhancedFREDClient

def debug_data_structure():
    """Debug the data structure and values"""
    
    api_key = "acf8bbec7efe3b6dfa6ae083e7152314"
    
    print("=== DEBUGGING DATA STRUCTURE ===")
    
    try:
        # Initialize FRED client
        client = EnhancedFREDClient(api_key)
        
        # Fetch economic data
        end_date = datetime.now()
        start_date = end_date.replace(year=end_date.year - 1)
        
        print("1. Fetching economic data...")
        data = client.fetch_economic_data(
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )
        
        if data.empty:
            print("❌ No data fetched")
            return
        
        print(f"✅ Fetched data shape: {data.shape}")
        print(f"   Date range: {data.index.min()} to {data.index.max()}")
        print(f"   Columns: {list(data.columns)}")
        print()
        
        # Check each indicator
        for column in data.columns:
            series = data[column].dropna()
            print(f"2. Analyzing {column}:")
            print(f"   Total observations: {len(data[column])}")
            print(f"   Non-null observations: {len(series)}")
            print(f"   Latest value: {series.iloc[-1] if len(series) > 0 else 'N/A'}")
            
            if len(series) >= 2:
                growth_rate = series.pct_change().iloc[-1] * 100
                print(f"   Latest growth rate: {growth_rate:.2f}%")
            else:
                print(f"   Growth rate: Insufficient data")
            
            if len(series) >= 13:
                yoy_growth = series.pct_change(periods=12).iloc[-1] * 100
                print(f"   Year-over-year growth: {yoy_growth:.2f}%")
            else:
                print(f"   Year-over-year growth: Insufficient data")
            
            print()
        
        # Test the specific calculations that are failing
        print("3. Testing specific calculations:")
        
        if 'GDPC1' in data.columns:
            gdp_series = data['GDPC1'].dropna()
            print(f"   GDPC1 - Length: {len(gdp_series)}")
            if len(gdp_series) >= 2:
                gdp_growth = gdp_series.pct_change().iloc[-1] * 100
                print(f"   GDPC1 - Growth: {gdp_growth:.2f}%")
                print(f"   GDPC1 - Is NaN: {pd.isna(gdp_growth)}")
            else:
                print(f"   GDPC1 - Insufficient data for growth calculation")
        
        if 'INDPRO' in data.columns:
            indpro_series = data['INDPRO'].dropna()
            print(f"   INDPRO - Length: {len(indpro_series)}")
            if len(indpro_series) >= 2:
                indpro_growth = indpro_series.pct_change().iloc[-1] * 100
                print(f"   INDPRO - Growth: {indpro_growth:.2f}%")
                print(f"   INDPRO - Is NaN: {pd.isna(indpro_growth)}")
            else:
                print(f"   INDPRO - Insufficient data for growth calculation")
        
        if 'CPIAUCSL' in data.columns:
            cpi_series = data['CPIAUCSL'].dropna()
            print(f"   CPIAUCSL - Length: {len(cpi_series)}")
            if len(cpi_series) >= 13:
                cpi_growth = cpi_series.pct_change(periods=12).iloc[-1] * 100
                print(f"   CPIAUCSL - YoY Growth: {cpi_growth:.2f}%")
                print(f"   CPIAUCSL - Is NaN: {pd.isna(cpi_growth)}")
            else:
                print(f"   CPIAUCSL - Insufficient data for YoY calculation")
        
        if 'FEDFUNDS' in data.columns:
            fed_series = data['FEDFUNDS'].dropna()
            print(f"   FEDFUNDS - Length: {len(fed_series)}")
            if len(fed_series) >= 1:
                fed_rate = fed_series.iloc[-1]
                print(f"   FEDFUNDS - Latest rate: {fed_rate:.2f}%")
                print(f"   FEDFUNDS - Is NaN: {pd.isna(fed_rate)}")
            else:
                print(f"   FEDFUNDS - No data available")
        
        if 'UNRATE' in data.columns:
            unrate_series = data['UNRATE'].dropna()
            print(f"   UNRATE - Length: {len(unrate_series)}")
            if len(unrate_series) >= 1:
                unrate = unrate_series.iloc[-1]
                print(f"   UNRATE - Latest rate: {unrate:.2f}%")
                print(f"   UNRATE - Is NaN: {pd.isna(unrate)}")
            else:
                print(f"   UNRATE - No data available")
        
        print()
        print("=== DEBUG COMPLETE ===")
        
    except Exception as e:
        print(f"❌ Error during debugging: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_data_structure() 