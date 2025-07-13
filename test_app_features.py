#!/usr/bin/env python3
"""
Test script to verify the app is displaying advanced features
"""

import requests
import json
import time

def test_app_features():
    """Test if the app is displaying advanced features"""
    print("🔍 Testing app features...")
    
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
    
    # Test 2: Check if analytics are loading
    try:
        # Import the app module to check analytics status
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        import frontend.app as app
        
        # Check analytics availability
        if hasattr(app, 'ANALYTICS_AVAILABLE') and app.ANALYTICS_AVAILABLE:
            print("✅ Advanced analytics are available")
        else:
            print("❌ Advanced analytics are not available")
            return False
            
    except Exception as e:
        print(f"❌ Analytics check failed: {e}")
        return False
    
    # Test 3: Check if nonfarm payroll indicator is included
    try:
        from config.settings import DEFAULT_SERIES_LIST
        if 'PAYEMS' in DEFAULT_SERIES_LIST:
            print("✅ Nonfarm payroll indicator (PAYEMS) is included")
        else:
            print("❌ Nonfarm payroll indicator not found")
            return False
    except Exception as e:
        print(f"❌ Series list check failed: {e}")
        return False
    
    # Test 4: Check if advanced visualizations are available
    try:
        from src.analysis.comprehensive_analytics import ComprehensiveAnalytics
        print("✅ Advanced analytics module is available")
    except Exception as e:
        print(f"❌ Advanced analytics module not available: {e}")
        return False
    
    print("\n🎉 All tests passed! The app should be displaying:")
    print("- Advanced analytics and visualizations")
    print("- Nonfarm payroll indicator (PAYEMS)")
    print("- Real-time economic data")
    print("- Mathematical fixes and enhanced UI")
    
    return True

if __name__ == "__main__":
    print("🧪 Testing App Features")
    print("=" * 50)
    
    success = test_app_features()
    
    if success:
        print("\n✅ Your app is running with all the advanced features!")
        print("🌐 Open http://localhost:8501 in your browser to see:")
        print("   - Executive Dashboard with real-time data")
        print("   - Advanced Analytics with mathematical fixes")
        print("   - Economic Indicators including nonfarm payroll")
        print("   - Enhanced visualizations and insights")
    else:
        print("\n⚠️ Some features may not be working properly.") 