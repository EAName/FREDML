#!/usr/bin/env python3
"""
Test script to verify analytics are loading after config fix
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_config_import():
    """Test if config.settings can be imported"""
    print("🔍 Testing config.settings import...")
    try:
        from config.settings import Config
        print("✅ Config import successful")
        return True
    except Exception as e:
        print(f"❌ Config import failed: {e}")
        return False

def test_analytics_import():
    """Test if analytics modules can be imported"""
    print("🔍 Testing analytics import...")
    try:
        from src.analysis.comprehensive_analytics import ComprehensiveAnalytics
        print("✅ Analytics import successful")
        return True
    except Exception as e:
        print(f"❌ Analytics import failed: {e}")
        return False

def test_app_analytics():
    """Test if the app can load analytics"""
    print("🔍 Testing app analytics loading...")
    try:
        # Import the app's analytics loading function
        import frontend.app as app
        
        # Check if analytics are available
        if hasattr(app, 'ANALYTICS_AVAILABLE'):
            print(f"✅ Analytics available: {app.ANALYTICS_AVAILABLE}")
            return app.ANALYTICS_AVAILABLE
        else:
            print("❌ ANALYTICS_AVAILABLE not found in app")
            return False
    except Exception as e:
        print(f"❌ App analytics test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Analytics Fix")
    print("=" * 50)
    
    config_ok = test_config_import()
    analytics_ok = test_analytics_import()
    app_analytics_ok = test_app_analytics()
    
    print("\n📊 Results:")
    print(f"Config Import: {'✅' if config_ok else '❌'}")
    print(f"Analytics Import: {'✅' if analytics_ok else '❌'}")
    print(f"App Analytics: {'✅' if app_analytics_ok else '❌'}")
    
    if config_ok and analytics_ok and app_analytics_ok:
        print("\n🎉 All tests passed! Analytics should be working.")
    else:
        print("\n⚠️ Some tests failed. Analytics may not be fully functional.") 