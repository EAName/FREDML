#!/usr/bin/env python3
"""
Test script to verify FRED API key functionality
"""

from fredapi import Fred
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.settings import FRED_API_KEY

def test_api_connection():
    """Test the FRED API connection with the provided key."""
    print("Testing FRED API connection...")
    
    try:
        # Initialize FRED client
        fred = Fred(api_key=FRED_API_KEY)
        
        # Test with a simple series (GDP)
        print("Fetching GDP data as a test...")
        gdp_data = fred.get_series('GDP', start='2023-01-01', end='2023-12-31')
        
        if not gdp_data.empty:
            print("✓ API connection successful!")
            print(f"✓ Retrieved {len(gdp_data)} GDP observations")
            print(f"✓ Latest GDP value: ${gdp_data.iloc[-1]:,.2f} billion")
            print(f"✓ Date range: {gdp_data.index.min()} to {gdp_data.index.max()}")
            return True
        else:
            print("✗ No data retrieved")
            return False
            
    except Exception as e:
        print(f"✗ API connection failed: {e}")
        return False

def test_series_info():
    """Test getting series information."""
    print("\nTesting series information retrieval...")
    
    try:
        fred = Fred(api_key=FRED_API_KEY)
        
        # Test getting info for GDP
        series_info = fred.get_series_info('GDP')
        
        print("✓ Series information retrieved successfully!")
        print(f"  Title: {series_info.title}")
        print(f"  Units: {series_info.units}")
        print(f"  Frequency: {series_info.frequency}")
        print(f"  Last Updated: {series_info.last_updated}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to get series info: {e}")
        return False

def main():
    """Run API tests."""
    print("FRED API Key Test")
    print("=" * 30)
    print(f"API Key: {FRED_API_KEY[:8]}...")
    print()
    
    # Test connection
    connection_ok = test_api_connection()
    
    # Test series info
    info_ok = test_series_info()
    
    print("\n" + "=" * 30)
    if connection_ok and info_ok:
        print("✓ All tests passed! Your API key is working correctly.")
        print("You can now use the FRED data collector tool.")
    else:
        print("✗ Some tests failed. Please check your API key.")
    
    return connection_ok and info_ok

if __name__ == "__main__":
    main() 