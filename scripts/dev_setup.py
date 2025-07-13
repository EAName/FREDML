#!/usr/bin/env python3
"""
FRED ML Development Environment Setup
Simple setup script for development testing
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    if version.major != 3 or version.minor < 9:
        print(f"❌ Python 3.9+ required, found {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_environment_variables():
    """Check required environment variables"""
    required_vars = ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'FRED_API_KEY']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        print("Please set these variables:")
        for var in missing_vars:
            print(f"   export {var}=your_value")
        return False
    
    print("✅ Environment variables set")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True, text=True)
        print("✅ Dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def test_imports():
    """Test that all required packages can be imported"""
    required_packages = [
        'boto3', 'streamlit', 'pandas', 'numpy', 'matplotlib', 
        'seaborn', 'plotly', 'fredapi', 'requests'
    ]
    
    failed_imports = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            failed_imports.append(package)
            print(f"❌ {package}")
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        return False
    
    return True

def test_aws_access():
    """Test AWS access"""
    try:
        import boto3
        s3 = boto3.client('s3')
        s3.head_bucket(Bucket='fredmlv1')
        print("✅ AWS S3 access")
        return True
    except Exception as e:
        print(f"❌ AWS S3 access failed: {str(e)}")
        return False

def test_fred_api():
    """Test FRED API access"""
    try:
        from fredapi import Fred
        fred = Fred(api_key=os.getenv('FRED_API_KEY'))
        data = fred.get_series('GDP', limit=1)
        if len(data) > 0:
            print("✅ FRED API access")
            return True
        else:
            print("❌ FRED API returned no data")
            return False
    except Exception as e:
        print(f"❌ FRED API access failed: {str(e)}")
        return False

def main():
    """Main setup function"""
    print("🚀 FRED ML Development Environment Setup")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version),
        ("Environment Variables", check_environment_variables),
        ("Dependencies", install_dependencies),
        ("Package Imports", test_imports),
        ("AWS Access", test_aws_access),
        ("FRED API", test_fred_api)
    ]
    
    passed = 0
    total = len(checks)
    
    for name, check_func in checks:
        print(f"\n🔍 Checking {name}...")
        if check_func():
            passed += 1
        else:
            print(f"❌ {name} check failed")
    
    print(f"\n📊 Setup Summary: {passed}/{total} checks passed")
    
    if passed == total:
        print("✅ Development environment ready!")
        print("\n🎯 Next steps:")
        print("1. Test the Streamlit app: streamlit run frontend/app.py")
        print("2. Test Lambda function: python scripts/test_complete_system.py")
        print("3. Run end-to-end tests: python scripts/test_complete_system.py --e2e")
        return True
    else:
        print("❌ Setup incomplete. Please fix the issues above.")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1) 