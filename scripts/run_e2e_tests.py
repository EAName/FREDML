#!/usr/bin/env python3
"""
End-to-End Test Runner for FRED ML
Runs comprehensive tests of the complete system
"""

import os
import sys
import subprocess
import argparse
import json
from pathlib import Path
import boto3
import time

def check_prerequisites():
    """Check if all prerequisites are met for testing"""
    print("🔍 Checking prerequisites...")
    
    # Check Python version
    if sys.version_info < (3, 9):
        print("❌ Python 3.9+ is required")
        return False
    
    # Check required packages
    required_packages = ['pytest', 'boto3', 'pandas', 'numpy']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    # Check AWS credentials
    try:
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        print(f"✅ AWS credentials configured for: {identity['Account']}")
    except Exception as e:
        print(f"❌ AWS credentials not configured: {e}")
        return False
    
    # Check AWS CLI
    try:
        subprocess.run(['aws', '--version'], capture_output=True, check=True)
        print("✅ AWS CLI found")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ AWS CLI not found")
        return False
    
    print("✅ All prerequisites met")
    return True

def setup_test_environment():
    """Set up test environment"""
    print("\n🔧 Setting up test environment...")
    
    # Set environment variables for testing
    os.environ['AWS_DEFAULT_REGION'] = 'us-west-2'
    os.environ['S3_BUCKET'] = 'fredmlv1'
    os.environ['LAMBDA_FUNCTION'] = 'fred-ml-processor'
    
    print("✅ Test environment configured")

def run_unit_tests():
    """Run unit tests"""
    print("\n🧪 Running unit tests...")
    
    try:
        result = subprocess.run([
            sys.executable, '-m', 'pytest', 
            'tests/unit/', 
            '-v', 
            '--tb=short'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Unit tests passed")
            return True
        else:
            print("❌ Unit tests failed")
            print(result.stdout)
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Unit test execution failed: {e}")
        return False

def run_integration_tests():
    """Run integration tests"""
    print("\n🔗 Running integration tests...")
    
    try:
        result = subprocess.run([
            sys.executable, '-m', 'pytest', 
            'tests/integration/', 
            '-v', 
            '--tb=short'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Integration tests passed")
            return True
        else:
            print("❌ Integration tests failed")
            print(result.stdout)
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Integration test execution failed: {e}")
        return False

def run_e2e_tests():
    """Run end-to-end tests"""
    print("\n🚀 Running end-to-end tests...")
    
    try:
        result = subprocess.run([
            sys.executable, '-m', 'pytest', 
            'tests/e2e/test_complete_workflow.py', 
            '-v', 
            '--tb=short',
            '--disable-warnings'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ End-to-end tests passed")
            return True
        else:
            print("❌ End-to-end tests failed")
            print(result.stdout)
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ End-to-end test execution failed: {e}")
        return False

def test_lambda_function_directly():
    """Test Lambda function directly (local simulation)"""
    print("\n⚡ Testing Lambda function directly...")
    
    try:
        # Import Lambda function
        sys.path.append(str(Path(__file__).parent.parent / 'lambda'))
        from lambda_function import lambda_handler
        
        # Test payload
        test_event = {
            'indicators': ['GDP'],
            'start_date': '2024-01-01',
            'end_date': '2024-01-31',
            'options': {
                'visualizations': False,
                'correlation': False,
                'statistics': True
            }
        }
        
        # Mock context
        class MockContext:
            def __init__(self):
                self.function_name = 'fred-ml-processor'
                self.function_version = '$LATEST'
                self.invoked_function_arn = 'arn:aws:lambda:us-west-2:123456789012:function:fred-ml-processor'
                self.memory_limit_in_mb = 512
                self.remaining_time_in_millis = 300000
                self.log_group_name = '/aws/lambda/fred-ml-processor'
                self.log_stream_name = '2024/01/01/[$LATEST]123456789012'
        
        context = MockContext()
        
        # Test function
        response = lambda_handler(test_event, context)
        
        if response.get('statusCode') == 200:
            print("✅ Lambda function test passed")
            return True
        else:
            print(f"❌ Lambda function test failed: {response}")
            return False
            
    except Exception as e:
        print(f"❌ Lambda function test failed: {e}")
        return False

def test_streamlit_app_locally():
    """Test Streamlit app locally"""
    print("\n🎨 Testing Streamlit app locally...")
    
    try:
        # Test Streamlit app imports
        sys.path.append(str(Path(__file__).parent.parent / 'frontend'))
        from app import load_config, init_aws_clients
        
        # Test configuration
        config = load_config()
        assert config['s3_bucket'] == 'fredmlv1'
        assert config['lambda_function'] == 'fred-ml-processor'
        print("✅ Streamlit configuration test passed")
        
        # Test AWS clients
        s3_client, lambda_client = init_aws_clients()
        if s3_client and lambda_client:
            print("✅ AWS clients initialization test passed")
        else:
            print("❌ AWS clients initialization failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Streamlit app test failed: {e}")
        return False

def generate_test_report(results):
    """Generate test report"""
    print("\n📊 Test Results Summary")
    print("=" * 50)
    
    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result)
    failed_tests = total_tests - passed_tests
    
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    print("\nDetailed Results:")
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    # Save report to file
    report_data = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_tests': total_tests,
        'passed_tests': passed_tests,
        'failed_tests': failed_tests,
        'success_rate': (passed_tests/total_tests)*100,
        'results': results
    }
    
    report_file = Path(__file__).parent.parent / 'test_report.json'
    with open(report_file, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: {report_file}")
    
    return passed_tests == total_tests

def main():
    parser = argparse.ArgumentParser(description='Run FRED ML End-to-End Tests')
    parser.add_argument('--skip-unit', action='store_true', help='Skip unit tests')
    parser.add_argument('--skip-integration', action='store_true', help='Skip integration tests')
    parser.add_argument('--skip-e2e', action='store_true', help='Skip end-to-end tests')
    parser.add_argument('--local-only', action='store_true', help='Run only local tests')
    
    args = parser.parse_args()
    
    print("🚀 FRED ML End-to-End Test Suite")
    print("=" * 50)
    
    # Check prerequisites
    if not check_prerequisites():
        print("❌ Prerequisites not met. Exiting.")
        sys.exit(1)
    
    # Setup environment
    setup_test_environment()
    
    # Run tests
    results = {}
    
    if not args.skip_unit:
        results['Unit Tests'] = run_unit_tests()
    
    if not args.skip_integration:
        results['Integration Tests'] = run_integration_tests()
    
    if not args.skip_e2e:
        results['End-to-End Tests'] = run_e2e_tests()
    
    if args.local_only:
        results['Lambda Function Test'] = test_lambda_function_directly()
        results['Streamlit App Test'] = test_streamlit_app_locally()
    
    # Generate report
    if results:
        success = generate_test_report(results)
        
        if success:
            print("\n🎉 All tests passed!")
            sys.exit(0)
        else:
            print("\n❌ Some tests failed. Check the report for details.")
            sys.exit(1)
    else:
        print("❌ No tests were run.")
        sys.exit(1)

if __name__ == "__main__":
    main() 