#!/usr/bin/env python3
"""
Simple Local App Test
Quick test to verify the app is working locally
"""

import requests
import os

def test_app_running():
    """Test if app is running"""
    print("🔍 Testing if app is running...")
    try:
        response = requests.get("http://localhost:8501/_stcore/health", timeout=5)
        if response.status_code == 200:
            print("✅ App is running on http://localhost:8501")
            return True
        else:
            print(f"❌ App health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ App not accessible: {e}")
        return False

def test_fred_api_key():
    """Test FRED API key configuration"""
    print("🔍 Testing FRED API key...")
    fred_key = os.getenv('FRED_API_KEY')
    if fred_key and fred_key != "your-fred-api-key-here":
        print("✅ FRED API key is configured")
        return True
    else:
        print("⚠️ FRED API key not configured (will use demo mode)")
        return False

def test_file_updates():
    """Test that key files have been updated"""
    print("🔍 Testing file updates...")
    
    # Check version in app.py
    with open("frontend/app.py", "r") as f:
        content = f.read()
        if "VERSION: 2.0.1" in content:
            print("✅ Version 2.0.1 found in app.py")
        else:
            print("❌ Version 2.0.1 not found in app.py")
            return False
    
    # Check Apache license
    with open("LICENSE", "r") as f:
        content = f.read()
        if "Apache License" in content and "Version 2.0" in content:
            print("✅ Apache 2.0 license applied")
        else:
            print("❌ Apache 2.0 license not found")
            return False
    
    # Check README
    with open("README.md", "r") as f:
        content = f.read()
        if "FRED ML - Federal Reserve Economic Data Machine Learning System" in content:
            print("✅ README has been updated")
        else:
            print("❌ README updates not found")
            return False
    
    return True

def main():
    """Run simple tests"""
    print("🚀 Starting Simple Local Tests...")
    print("=" * 40)
    
    tests = [
        test_app_running,
        test_fred_api_key,
        test_file_updates
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed: {e}")
        print()
    
    print("=" * 40)
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        print("\n✅ Local app is working correctly")
        print("🌐 Open http://localhost:8501 in your browser")
        print("\n📋 What to check manually:")
        print("  - Look for 'FRED ML v2.0.1' banner at the top")
        print("  - Check that all pages load without errors")
        print("  - Verify economic data is displayed")
        print("  - Test downloads section functionality")
    else:
        print("⚠️ Some tests failed. Check the output above.")
    
    print(f"\n🌐 Local App URL: http://localhost:8501")

if __name__ == "__main__":
    main() 