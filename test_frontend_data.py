#!/usr/bin/env python3
"""
Test script to check what the frontend FRED client returns
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Add frontend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'frontend'))

from frontend.fred_api_client import get_real_economic_data

def test_frontend_data():
    """Test what the frontend client returns"""
    
    api_key = "acf8bbec7efe3b6dfa6ae083e7152314"
    
    print("=== TESTING FRONTEND FRED CLIENT ===")
    
    try:
        # Get data using frontend client
        end_date = datetime.now()
        start_date = end_date.replace(year=end_date.year - 1)
        
        print("1. Fetching data with frontend client...")
        real_data = get_real_economic_data(
            api_key, 
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        print(f"✅ Real data keys: {list(real_data.keys())}")
        
        # Check economic_data
        if 'economic_data' in real_data:
            df = real_data['economic_data']
            print(f"   Economic data shape: {df.shape}")
            print(f"   Economic data columns: {list(df.columns)}")
            print(f"   Economic data index: {df.index.min()} to {df.index.max()}")
            
            if not df.empty:
                print("   Sample data:")
                print(df.head())
                print()
                
                # Test calculations
                print("2. Testing calculations on frontend data:")
                
                for column in df.columns:
                    series = df[column].dropna()
                    print(f"   {column}:")
                    print(f"     Length: {len(series)}")
                    print(f"     Latest value: {series.iloc[-1] if len(series) > 0 else 'N/A'}")
                    
                    if len(series) >= 2:
                        growth_rate = series.pct_change().iloc[-1] * 100
                        print(f"     Growth rate: {growth_rate:.2f}%")
                        print(f"     Is NaN: {pd.isna(growth_rate)}")
                    else:
                        print(f"     Growth rate: Insufficient data")
                    print()
            else:
                print("   ❌ Economic data is empty!")
        else:
            print("   ❌ No economic_data in real_data")
        
        # Check insights
        if 'insights' in real_data:
            insights = real_data['insights']
            print(f"   Insights keys: {list(insights.keys())}")
            
            # Show some sample insights
            for series_id, insight in list(insights.items())[:3]:
                print(f"   {series_id}:")
                print(f"     Current value: {insight.get('current_value', 'N/A')}")
                print(f"     Growth rate: {insight.get('growth_rate', 'N/A')}")
                print(f"     Trend: {insight.get('trend', 'N/A')}")
                print()
        else:
            print("   ❌ No insights in real_data")
        
        print("=== FRONTEND CLIENT TEST COMPLETE ===")
        
    except Exception as e:
        print(f"❌ Error testing frontend client: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_frontend_data() 