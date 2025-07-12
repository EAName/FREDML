#!/usr/bin/env python3
"""
FRED ML - Setup Script
Help users set up their FRED API key and test the connection
"""

import os
import sys
from pathlib import Path

def create_env_file():
    """Create a .env file with FRED API key template"""
    env_file = Path(".env")
    
    if env_file.exists():
        print("📄 .env file already exists")
        return False
    
    env_content = """# FRED ML Environment Configuration
# Get your free API key from: https://fred.stlouisfed.org/docs/api/api_key.html

FRED_API_KEY=your-fred-api-key-here

# AWS Configuration (optional)
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key

# Application Settings
LOG_LEVEL=INFO
ENVIRONMENT=development
"""
    
    try:
        with open(env_file, 'w') as f:
            f.write(env_content)
        print("✅ Created .env file with template")
        return True
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = ['requests', 'pandas', 'streamlit']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Install them with: pip install -r requirements.txt")
        return False
    else:
        print("✅ All required packages are installed")
        return True

def main():
    """Main setup function"""
    print("=" * 60)
    print("FRED ML - Setup Wizard")
    print("=" * 60)
    
    # Check dependencies
    print("\n🔍 Checking dependencies...")
    if not check_dependencies():
        return False
    
    # Create .env file
    print("\n📄 Setting up environment file...")
    create_env_file()
    
    # Instructions
    print("\n📋 Next Steps:")
    print("1. Get a free FRED API key from: https://fred.stlouisfed.org/docs/api/api_key.html")
    print("2. Edit the .env file and replace 'your-fred-api-key-here' with your actual API key")
    print("3. Test your API key: python frontend/test_fred_api.py")
    print("4. Run the application: cd frontend && streamlit run app.py")
    
    print("\n" + "=" * 60)
    print("🎉 Setup complete!")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 