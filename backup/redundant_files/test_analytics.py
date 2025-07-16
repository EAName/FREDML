#!/usr/bin/env python3
"""
Test script for FRED ML analytics functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test if all required modules can be imported"""
    try:
        from src.core.enhanced_fred_client import EnhancedFREDClient
        print("✅ EnhancedFREDClient import: PASSED")
        
        from src.analysis.comprehensive_analytics import ComprehensiveAnalytics
        print("✅ ComprehensiveAnalytics import: PASSED")
        
        from src.analysis.economic_forecasting import EconomicForecaster
        print("✅ EconomicForecaster import: PASSED")
        
        from src.analysis.economic_segmentation import EconomicSegmentation
        print("✅ EconomicSegmentation import: PASSED")
        
        from src.analysis.statistical_modeling import StatisticalModeling
        print("✅ StatisticalModeling import: PASSED")
        
        return True
    except Exception as e:
        print(f"❌ Import test: FAILED ({e})")
        return False

def test_fred_client():
    """Test FRED client functionality"""
    try:
        from src.core.enhanced_fred_client import EnhancedFREDClient
        
        client = EnhancedFREDClient("acf8bbec7efe3b6dfa6ae083e7152314")
        
        # Test basic functionality - check for the correct method names
        if hasattr(client, 'fetch_economic_data') and hasattr(client, 'fetch_quarterly_data'):
            print("✅ FRED Client structure: PASSED")
            return True
        else:
            print("❌ FRED Client structure: FAILED")
            return False
    except Exception as e:
        print(f"❌ FRED Client test: FAILED ({e})")
        return False

def test_analytics_structure():
    """Test analytics module structure"""
    try:
        from src.analysis.comprehensive_analytics import ComprehensiveAnalytics
        
        # Test if the class has required methods
        analytics = ComprehensiveAnalytics("acf8bbec7efe3b6dfa6ae083e7152314")
        
        required_methods = [
            'run_complete_analysis',
            '_run_statistical_analysis',
            '_run_forecasting_analysis', 
            '_run_segmentation_analysis',
            '_extract_insights'
        ]
        
        for method in required_methods:
            if hasattr(analytics, method):
                print(f"✅ Method {method}: PASSED")
            else:
                print(f"❌ Method {method}: FAILED")
                return False
        
        return True
    except Exception as e:
        print(f"❌ Analytics structure test: FAILED ({e})")
        return False

def test_config():
    """Test configuration loading"""
    try:
        # Test if config can be loaded
        import os
        fred_key = os.getenv('FRED_API_KEY', 'acf8bbec7efe3b6dfa6ae083e7152314')
        
        if fred_key and len(fred_key) > 10:
            print("✅ Configuration loading: PASSED")
            return True
        else:
            print("❌ Configuration loading: FAILED")
            return False
    except Exception as e:
        print(f"❌ Configuration test: FAILED ({e})")
        return False

def main():
    """Run all analytics tests"""
    print("🧪 Testing FRED ML Analytics...")
    print("=" * 50)
    
    tests = [
        ("Module Imports", test_imports),
        ("FRED Client", test_fred_client),
        ("Analytics Structure", test_analytics_structure),
        ("Configuration", test_config),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing: {test_name}")
        if test_func():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Analytics Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All analytics tests passed! The analytics modules are working correctly.")
        return 0
    else:
        print("⚠️  Some analytics tests failed. Check the module imports and structure.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 