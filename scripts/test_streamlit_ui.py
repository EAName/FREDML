#!/usr/bin/env python3
"""
FRED ML - Streamlit UI Test
Simple test to validate Streamlit UI functionality
"""

import os
import sys
import subprocess
from pathlib import Path

def test_streamlit_ui():
    """Test Streamlit UI functionality"""
    print("🎨 Testing Streamlit UI...")
    
    # Check if Streamlit app exists
    app_path = Path(__file__).parent.parent / 'frontend/app.py'
    if not app_path.exists():
        print("❌ Streamlit app not found")
        return False
    
    print("✅ Streamlit app exists")
    
    # Check app content
    with open(app_path, 'r') as f:
        content = f.read()
    
    # Check for required components
    required_components = [
        'st.set_page_config',
        'show_executive_dashboard',
        'show_advanced_analytics_page',
        'show_indicators_page',
        'show_reports_page',
        'show_configuration_page'
    ]
    
    missing_components = []
    for component in required_components:
        if component not in content:
            missing_components.append(component)
    
    if missing_components:
        print(f"❌ Missing components in Streamlit app: {missing_components}")
        return False
    else:
        print("✅ All required Streamlit components found")
    
    # Check for enterprise styling
    styling_components = [
        'main-header',
        'metric-card',
        'analysis-section',
        'chart-container'
    ]
    
    missing_styling = []
    for component in styling_components:
        if component not in content:
            missing_styling.append(component)
    
    if missing_styling:
        print(f"⚠️ Missing styling components: {missing_styling}")
    else:
        print("✅ Enterprise styling components found")
    
    # Check for analytics integration
    analytics_components = [
        'ComprehensiveAnalytics',
        'EnhancedFREDClient',
        'display_analysis_results'
    ]
    
    missing_analytics = []
    for component in analytics_components:
        if component not in content:
            missing_analytics.append(component)
    
    if missing_analytics:
        print(f"⚠️ Missing analytics components: {missing_analytics}")
    else:
        print("✅ Analytics integration components found")
    
    print("✅ Streamlit UI test passed")
    return True

def test_streamlit_syntax():
    """Test Streamlit app syntax"""
    print("🔍 Testing Streamlit app syntax...")
    
    app_path = Path(__file__).parent.parent / 'frontend/app.py'
    
    try:
        with open(app_path, 'r') as f:
            compile(f.read(), str(app_path), 'exec')
        print("✅ Streamlit app syntax is valid")
        return True
    except SyntaxError as e:
        print(f"❌ Streamlit app syntax error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing syntax: {e}")
        return False

def test_streamlit_launch():
    """Test if Streamlit can launch the app"""
    print("🚀 Testing Streamlit launch capability...")
    
    try:
        # Test if streamlit is available
        result = subprocess.run(
            ['streamlit', '--version'],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print(f"✅ Streamlit version: {result.stdout.strip()}")
            return True
        else:
            print("❌ Streamlit not available")
            return False
            
    except FileNotFoundError:
        print("❌ Streamlit not installed")
        return False
    except Exception as e:
        print(f"❌ Error testing Streamlit: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Starting Streamlit UI Test")
    print("=" * 50)
    
    # Test 1: UI Components
    ui_test = test_streamlit_ui()
    
    # Test 2: Syntax
    syntax_test = test_streamlit_syntax()
    
    # Test 3: Launch capability
    launch_test = test_streamlit_launch()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 STREAMLIT UI TEST RESULTS")
    print("=" * 50)
    
    tests = [
        ("UI Components", ui_test),
        ("Syntax Check", syntax_test),
        ("Launch Capability", launch_test)
    ]
    
    passed = 0
    for test_name, result in tests:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All Streamlit UI tests passed!")
        return True
    else:
        print("❌ Some Streamlit UI tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 