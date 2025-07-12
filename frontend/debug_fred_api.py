#!/usr/bin/env python3
"""
FRED ML - Debug FRED API Issues
Debug specific series that are failing
"""

import os
import requests
import json

def debug_series(series_id: str, api_key: str):
    """Debug a specific series to see what's happening"""
    print(f"\n🔍 Debugging {series_id}...")
    
    try:
        # Test with a simple series request
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            'series_id': series_id,
            'api_key': api_key,
            'file_type': 'json',
            'limit': 5
        }
        
        print(f"URL: {url}")
        print(f"Params: {params}")
        
        response = requests.get(url, params=params)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Response Data: {json.dumps(data, indent=2)}")
            
            if 'observations' in data:
                print(f"Number of observations: {len(data['observations'])}")
                if len(data['observations']) > 0:
                    print(f"First observation: {data['observations'][0]}")
                else:
                    print("No observations found")
            else:
                print("No 'observations' key in response")
        else:
            print(f"Error Response: {response.text}")
            
    except Exception as e:
        print(f"Exception: {e}")

def test_series_info(series_id: str, api_key: str):
    """Test series info endpoint"""
    print(f"\n📊 Testing series info for {series_id}...")
    
    try:
        url = "https://api.stlouisfed.org/fred/series"
        params = {
            'series_id': series_id,
            'api_key': api_key,
            'file_type': 'json'
        }
        
        response = requests.get(url, params=params)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Series Info: {json.dumps(data, indent=2)}")
        else:
            print(f"Error Response: {response.text}")
            
    except Exception as e:
        print(f"Exception: {e}")

def main():
    """Main debug function"""
    print("=" * 60)
    print("FRED ML - API Debug Tool")
    print("=" * 60)
    
    # Get API key from environment
    api_key = os.getenv('FRED_API_KEY')
    
    if not api_key:
        print("❌ FRED_API_KEY environment variable not set")
        return
    
    # Test problematic series
    problematic_series = ['FEDFUNDS', 'INDPRO']
    
    for series_id in problematic_series:
        debug_series(series_id, api_key)
        test_series_info(series_id, api_key)
    
    # Test with different parameters
    print("\n🔧 Testing with different parameters...")
    
    for series_id in problematic_series:
        print(f"\nTesting {series_id} with different limits...")
        
        for limit in [1, 5, 10]:
            try:
                url = "https://api.stlouisfed.org/fred/series/observations"
                params = {
                    'series_id': series_id,
                    'api_key': api_key,
                    'file_type': 'json',
                    'limit': limit
                }
                
                response = requests.get(url, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    obs_count = len(data.get('observations', []))
                    print(f"  Limit {limit}: {obs_count} observations")
                else:
                    print(f"  Limit {limit}: Failed with status {response.status_code}")
                    
            except Exception as e:
                print(f"  Limit {limit}: Exception - {e}")

if __name__ == "__main__":
    main() 