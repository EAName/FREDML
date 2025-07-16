#!/usr/bin/env python3
"""
Deployment Check Script for Streamlit Cloud
Verifies that all necessary files are present and properly configured
"""

import os
import sys

def check_deployment_files():
    """Check if all necessary files exist for Streamlit Cloud deployment"""
    
    print("🔍 Checking Streamlit Cloud Deployment Files...")
    
    # Check main app file
    if os.path.exists("frontend/app.py"):
        print("✅ frontend/app.py exists")
    else:
        print("❌ frontend/app.py missing")
        return False
    
    # Check requirements.txt
    if os.path.exists("requirements.txt"):
        print("✅ requirements.txt exists")
        with open("requirements.txt", "r") as f:
            requirements = f.read()
            if "streamlit" in requirements:
                print("✅ streamlit in requirements.txt")
            else:
                print("❌ streamlit missing from requirements.txt")
    else:
        print("❌ requirements.txt missing")
        return False
    
    # Check .gitignore
    if os.path.exists(".gitignore"):
        print("✅ .gitignore exists")
    else:
        print("⚠️ .gitignore missing (optional)")
    
    # Check for any large files that might cause issues
    large_files = []
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith(('.csv', '.json', '.png', '.jpg', '.jpeg')):
                filepath = os.path.join(root, file)
                size = os.path.getsize(filepath)
                if size > 10 * 1024 * 1024:  # 10MB
                    large_files.append((filepath, size))
    
    if large_files:
        print("⚠️ Large files detected (may cause deployment issues):")
        for filepath, size in large_files:
            print(f"   {filepath} ({size / 1024 / 1024:.1f}MB)")
    else:
        print("✅ No large files detected")
    
    # Check environment variables
    fred_key = os.getenv("FRED_API_KEY")
    if fred_key and fred_key != "your-fred-api-key-here":
        print("✅ FRED_API_KEY environment variable set")
    else:
        print("⚠️ FRED_API_KEY not set (will use demo mode)")
    
    print("\n📋 Streamlit Cloud Configuration Checklist:")
    print("1. Main file path: frontend/app.py")
    print("2. Git branch: main")
    print("3. Repository: ParallelLLC/FREDML")
    print("4. Environment variables: FRED_API_KEY")
    
    return True

if __name__ == "__main__":
    success = check_deployment_files()
    if success:
        print("\n✅ Deployment files look good!")
        print("If Streamlit Cloud still shows old version, try:")
        print("1. Force redeploy in Streamlit Cloud dashboard")
        print("2. Check deployment logs for errors")
        print("3. Verify branch and file path settings")
    else:
        print("\n❌ Deployment files need attention")
        sys.exit(1) 