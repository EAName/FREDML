#!/usr/bin/env python3
"""
Simple FRED API test
"""

import os
import sys

import requests

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from config.settings import FRED_API_KEY


def test_fred_api_direct():
    """Test FRED API directly using requests."""
    print("Testing FRED API directly...")

    # Test URL for GDP series
    url = f"https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": "GDP",
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "start_date": "2023-01-01",
        "end_date": "2023-12-31",
    }

    try:
        response = requests.get(url, params=params)

        if response.status_code == 200:
            data = response.json()
            observations = data.get("observations", [])

            if observations:
                print("✓ API connection successful!")
                print(f"✓ Retrieved {len(observations)} GDP observations")

                # Get the latest observation
                latest = observations[-1]
                print(f"✓ Latest GDP value: ${float(latest['value']):,.2f} billion")
                print(f"✓ Date: {latest['date']}")
                return True
            else:
                print("✗ No observations found")
                return False
        else:
            print(f"✗ API request failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
            return False

    except Exception as e:
        print(f"✗ API connection failed: {e}")
        return False


def test_series_search():
    """Test searching for series."""
    print("\nTesting series search...")

    url = "https://api.stlouisfed.org/fred/series/search"
    params = {"search_text": "GDP", "api_key": FRED_API_KEY, "file_type": "json"}

    try:
        response = requests.get(url, params=params)

        if response.status_code == 200:
            data = response.json()
            series = data.get("seriess", [])

            if series:
                print("✓ Series search successful!")
                print(f"✓ Found {len(series)} series matching 'GDP'")

                # Show first few results
                for i, s in enumerate(series[:3]):
                    print(f"  {i+1}. {s['id']}: {s['title']}")
                return True
            else:
                print("✗ No series found")
                return False
        else:
            print(f"✗ Search request failed: {response.status_code}")
            return False

    except Exception as e:
        print(f"✗ Search failed: {e}")
        return False


def main():
    """Run simple API tests."""
    print("Simple FRED API Test")
    print("=" * 30)
    print(f"API Key: {FRED_API_KEY[:8]}...")
    print()

    # Test direct API access
    api_ok = test_fred_api_direct()

    # Test series search
    search_ok = test_series_search()

    print("\n" + "=" * 30)
    if api_ok and search_ok:
        print("✓ All tests passed! Your API key is working correctly.")
        print("The issue is with the fredapi library, not your API key.")
    else:
        print("✗ Some tests failed. Please check your API key.")

    return api_ok and search_ok


if __name__ == "__main__":
    main()
