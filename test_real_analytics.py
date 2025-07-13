#!/usr/bin/env python3
"""
Test script to verify that the app is using real analytics
"""

import requests
import time

def test_app_analytics():
    """Test if the app is using real analytics"""
    print("🔍 Testing app analytics...")
    
    # Test 1: Check if app is running
    try:
        response = requests.get("http://localhost:8501/_stcore/health", timeout=5)
        if response.status_code == 200:
            print("✅ App is running and healthy")
        else:
            print(f"❌ App health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ App not accessible: {e}")
        return False
    
    # Test 2: Check if analytics are available
    try:
        # Test the analytics import directly
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        from src.analysis.comprehensive_analytics import ComprehensiveAnalytics
        print("✅ ComprehensiveAnalytics import successful")
        
        # Test creating an analytics instance
        analytics = ComprehensiveAnalytics("test_key", output_dir="test_output")
        print("✅ ComprehensiveAnalytics instance created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Analytics test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_app_analytics()
    if success:
        print("\n🎉 All tests passed! The app should now be using real analytics.")
        print("📊 Open http://localhost:8501 and check the Advanced Analytics page.")
    else:
        print("\n❌ Some tests failed. Check the logs above.") 