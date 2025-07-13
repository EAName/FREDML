#!/usr/bin/env python3
"""
Debug script to test analytics import
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test all the imports that the analytics need"""
    print("🔍 Testing imports...")
    
    # Test 1: Config import
    print("\n1. Testing config import...")
    try:
        from config.settings import Config
        print("✅ Config import successful")
        config = Config()
        print(f"✅ Config.get_fred_api_key() = {config.get_fred_api_key()}")
    except Exception as e:
        print(f"❌ Config import failed: {e}")
    
    # Test 2: Analytics import
    print("\n2. Testing analytics import...")
    try:
        from src.analysis.comprehensive_analytics import ComprehensiveAnalytics
        print("✅ ComprehensiveAnalytics import successful")
    except Exception as e:
        print(f"❌ ComprehensiveAnalytics import failed: {e}")
    
    # Test 3: Enhanced FRED Client import
    print("\n3. Testing Enhanced FRED Client import...")
    try:
        from src.core.enhanced_fred_client import EnhancedFREDClient
        print("✅ EnhancedFREDClient import successful")
    except Exception as e:
        print(f"❌ EnhancedFREDClient import failed: {e}")
    
    # Test 4: Economic Forecasting import
    print("\n4. Testing Economic Forecasting import...")
    try:
        from src.analysis.economic_forecasting import EconomicForecaster
        print("✅ EconomicForecaster import successful")
    except Exception as e:
        print(f"❌ EconomicForecaster import failed: {e}")
    
    # Test 5: Economic Segmentation import
    print("\n5. Testing Economic Segmentation import...")
    try:
        from src.analysis.economic_segmentation import EconomicSegmentation
        print("✅ EconomicSegmentation import successful")
    except Exception as e:
        print(f"❌ EconomicSegmentation import failed: {e}")
    
    # Test 6: Statistical Modeling import
    print("\n6. Testing Statistical Modeling import...")
    try:
        from src.analysis.statistical_modeling import StatisticalModeling
        print("✅ StatisticalModeling import successful")
    except Exception as e:
        print(f"❌ StatisticalModeling import failed: {e}")

if __name__ == "__main__":
    test_imports() 