#!/usr/bin/env python3
"""
Simple Test Runner for FRED ML
Run this script to test the complete system
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Run the complete system test"""
    print("🚀 FRED ML Complete System Test")
    print("=" * 50)
    
    # Check if the test script exists
    test_script = Path(__file__).parent / 'scripts' / 'test_complete_system.py'
    
    if not test_script.exists():
        print("❌ Test script not found. Please run the deployment first.")
        sys.exit(1)
    
    # Run the test
    try:
        result = subprocess.run([
            sys.executable, str(test_script)
        ], check=True)
        
        print("\n🎉 Test completed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Test failed with exit code: {e.returncode}")
        return False
    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 