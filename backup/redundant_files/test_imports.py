#!/usr/bin/env python3
"""
Test script to verify all analytics imports work correctly
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test all the imports that the analytics need"""
    print("🔍 Testing analytics imports...")
    
    # Test 1: Config import
    print("\n1. Testing config import...")
    try:
        from config.settings import Config
        print("✅ Config import successful")
        config = Config()
        print(f"✅ Config.get_fred_api_key() = {config.get_fred_api_key()}")
    except Exception as e:
        print(f"❌ Config import failed: {e}")
        return False
    
    # Test 2: Analytics import
    print("\n2. Testing analytics import...")
    try:
        from src.analysis.comprehensive_analytics import ComprehensiveAnalytics
        print("✅ ComprehensiveAnalytics import successful")
    except Exception as e:
        print(f"❌ ComprehensiveAnalytics import failed: {e}")
        return False
    
    # Test 3: FRED Client import
    print("\n3. Testing FRED client import...")
    try:
        from src.core.enhanced_fred_client import EnhancedFREDClient
        print("✅ EnhancedFREDClient import successful")
    except Exception as e:
        print(f"❌ EnhancedFREDClient import failed: {e}")
        return False
    
    # Test 4: Analytics modules import
    print("\n4. Testing analytics modules import...")
    try:
        from src.analysis.economic_forecasting import EconomicForecaster
        from src.analysis.economic_segmentation import EconomicSegmentation
        from src.analysis.statistical_modeling import StatisticalModeling
        print("✅ All analytics modules import successful")
    except Exception as e:
        print(f"❌ Analytics modules import failed: {e}")
        return False
    
    # Test 5: Create analytics instance
    print("\n5. Testing analytics instance creation...")
    try:
        analytics = ComprehensiveAnalytics(api_key="test_key", output_dir="test_output")
        print("✅ ComprehensiveAnalytics instance created successfully")
    except Exception as e:
        print(f"❌ Analytics instance creation failed: {e}")
        return False
    
    print("\n🎉 All imports and tests passed successfully!")
    return True

if __name__ == "__main__":
    success = test_imports()
    if success:
        print("\n✅ All analytics imports are working correctly!")
    else:
        print("\n❌ Some imports failed. Check the errors above.") 