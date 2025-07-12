#!/usr/bin/env python3
"""
FRED ML - FRED API Test Script
Test your FRED API connection and key
"""

import os
import sys
import requests
from datetime import datetime, timedelta

def test_fred_api_key(api_key: str) -> bool:
    """Test FRED API key by making a simple request"""
    try:
        # Test with a simple series request
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            'series_id': 'GDPC1',  # Real GDP
            'api_key': api_key,
            'file_type': 'json',
            'limit': 1
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            if 'observations' in data and len(data['observations']) > 0:
                print("✅ FRED API key is valid!")
                print(f"📊 Successfully fetched GDP data: {data['observations'][0]}")
                return True
            else:
                print("❌ API key may be invalid - no data returned")
                return False
        else:
            print(f"❌ API request failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing FRED API: {e}")
        return False

def test_multiple_series(api_key: str) -> bool:
    """Test multiple economic series"""
    series_list = [
        'GDPC1',    # Real GDP
        'INDPRO',   # Industrial Production
        'CPIAUCSL', # Consumer Price Index
        'FEDFUNDS', # Federal Funds Rate
        'DGS10',    # 10-Year Treasury
        'UNRATE'    # Unemployment Rate
    ]
    
    print("\n🔍 Testing multiple economic series...")
    
    for series_id in series_list:
        try:
            url = "https://api.stlouisfed.org/fred/series/observations"
            params = {
                'series_id': series_id,
                'api_key': api_key,
                'file_type': 'json',
                'limit': 5  # Use limit=5 to avoid timeout issues
            }
            
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                if 'observations' in data and len(data['observations']) > 0:
                    latest_value = data['observations'][-1]['value']  # Get the latest (last) observation
                    latest_date = data['observations'][-1]['date']
                    print(f"✅ {series_id}: {latest_value} ({latest_date})")
                else:
                    print(f"❌ {series_id}: No data available")
            else:
                print(f"❌ {series_id}: Request failed with status {response.status_code}")
                
        except Exception as e:
            print(f"❌ {series_id}: Error - {e}")
    
    return True

def main():
    """Main function to test FRED API"""
    print("=" * 60)
    print("FRED ML - API Key Test")
    print("=" * 60)
    
    # Get API key from environment
    api_key = os.getenv('FRED_API_KEY')
    
    if not api_key:
        print("❌ FRED_API_KEY environment variable not set")
        print("\nTo set it, run:")
        print("export FRED_API_KEY='your-api-key-here'")
        return False
    
    if api_key == 'your-fred-api-key-here':
        print("❌ Please replace 'your-fred-api-key-here' with your actual API key")
        return False
    
    print(f"🔑 Testing API key: {api_key[:8]}...")
    
    # Test basic API connection
    if test_fred_api_key(api_key):
        # Test multiple series
        test_multiple_series(api_key)
        
        print("\n" + "=" * 60)
        print("🎉 FRED API is working correctly!")
        print("✅ You can now use real economic data in the application")
        print("=" * 60)
        return True
    else:
        print("\n" + "=" * 60)
        print("❌ FRED API test failed")
        print("Please check your API key and try again")
        print("=" * 60)
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 