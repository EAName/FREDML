#!/usr/bin/env python3
"""
Test script for FRED ML app functionality
"""

import requests
import time
import sys

def test_app_health():
    """Test if the app is running and healthy"""
    try:
        response = requests.get("http://localhost:8501/_stcore/health", timeout=5)
        if response.status_code == 200:
            print("✅ App health check: PASSED")
            return True
        else:
            print(f"❌ App health check: FAILED (status {response.status_code})")
            return False
    except Exception as e:
        print(f"❌ App health check: FAILED ({e})")
        return False

def test_app_loading():
    """Test if the app loads the main page"""
    try:
        response = requests.get("http://localhost:8501", timeout=10)
        if response.status_code == 200 and "Streamlit" in response.text:
            print("✅ App main page: PASSED")
            return True
        else:
            print(f"❌ App main page: FAILED (status {response.status_code})")
            return False
    except Exception as e:
        print(f"❌ App main page: FAILED ({e})")
        return False

def test_fred_api():
    """Test FRED API functionality"""
    try:
        # Test FRED API key
        api_key = "acf8bbec7efe3b6dfa6ae083e7152314"
        test_url = f"https://api.stlouisfed.org/fred/series?series_id=GDP&api_key={api_key}&file_type=json"
        response = requests.get(test_url, timeout=10)
        if response.status_code == 200:
            print("✅ FRED API test: PASSED")
            return True
        else:
            print(f"❌ FRED API test: FAILED (status {response.status_code})")
            return False
    except Exception as e:
        print(f"❌ FRED API test: FAILED ({e})")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing FRED ML App...")
    print("=" * 50)
    
    tests = [
        ("App Health", test_app_health),
        ("App Loading", test_app_loading),
        ("FRED API", test_fred_api),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing: {test_name}")
        if test_func():
            passed += 1
        time.sleep(1)  # Brief pause between tests
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The app is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the logs for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 