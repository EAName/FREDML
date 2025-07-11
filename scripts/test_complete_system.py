#!/usr/bin/env python3
"""
Complete System Test for FRED ML
Tests the entire workflow: Streamlit → Lambda → S3 → Reports
"""

import os
import sys
import json
import time
import boto3
import subprocess
from pathlib import Path
from datetime import datetime, timedelta

def print_header(title):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f"🧪 {title}")
    print(f"{'='*60}")

def print_success(message):
    """Print success message"""
    print(f"✅ {message}")

def print_error(message):
    """Print error message"""
    print(f"❌ {message}")

def print_warning(message):
    """Print warning message"""
    print(f"⚠️  {message}")

def print_info(message):
    """Print info message"""
    print(f"ℹ️  {message}")

def check_prerequisites():
    """Check if all prerequisites are met"""
    print_header("Checking Prerequisites")
    
    # Check Python version
    if sys.version_info < (3, 9):
        print_error("Python 3.9+ is required")
        return False
    print_success(f"Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Check required packages
    required_packages = ['boto3', 'pandas', 'numpy', 'requests']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print_success(f"{package} is available")
        except ImportError:
            missing_packages.append(package)
            print_error(f"{package} is missing")
    
    if missing_packages:
        print_error(f"Missing packages: {', '.join(missing_packages)}")
        print_info("Run: pip install -r requirements.txt")
        return False
    
    # Check AWS credentials
    try:
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        print_success(f"AWS credentials configured for account: {identity['Account']}")
    except Exception as e:
        print_error(f"AWS credentials not configured: {e}")
        return False
    
    # Check AWS CLI
    try:
        result = subprocess.run(['aws', '--version'], capture_output=True, text=True, check=True)
        print_success("AWS CLI is available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print_warning("AWS CLI not found (optional)")
    
    return True

def test_aws_services():
    """Test AWS services connectivity"""
    print_header("Testing AWS Services")
    
    # Test S3
    try:
        s3 = boto3.client('s3', region_name='us-west-2')
        response = s3.head_bucket(Bucket='fredmlv1')
        print_success("S3 bucket 'fredmlv1' is accessible")
    except Exception as e:
        print_error(f"S3 bucket access failed: {e}")
        return False
    
    # Test Lambda
    try:
        lambda_client = boto3.client('lambda', region_name='us-west-2')
        response = lambda_client.get_function(FunctionName='fred-ml-processor')
        print_success("Lambda function 'fred-ml-processor' exists")
        print_info(f"Runtime: {response['Configuration']['Runtime']}")
        print_info(f"Memory: {response['Configuration']['MemorySize']} MB")
        print_info(f"Timeout: {response['Configuration']['Timeout']} seconds")
    except Exception as e:
        print_error(f"Lambda function not found: {e}")
        return False
    
    # Test SSM
    try:
        ssm = boto3.client('ssm', region_name='us-west-2')
        response = ssm.get_parameter(Name='/fred-ml/api-key', WithDecryption=True)
        api_key = response['Parameter']['Value']
        if api_key and api_key != 'your-fred-api-key-here':
            print_success("FRED API key is configured in SSM")
        else:
            print_error("FRED API key not properly configured")
            return False
    except Exception as e:
        print_error(f"SSM parameter not found: {e}")
        return False
    
    return True

def test_lambda_function():
    """Test Lambda function invocation"""
    print_header("Testing Lambda Function")
    
    try:
        lambda_client = boto3.client('lambda', region_name='us-west-2')
        
        # Test payload
        test_payload = {
            'indicators': ['GDP', 'UNRATE'],
            'start_date': '2024-01-01',
            'end_date': '2024-01-31',
            'options': {
                'visualizations': True,
                'correlation': True,
                'forecasting': False,
                'statistics': True
            }
        }
        
        print_info("Invoking Lambda function...")
        response = lambda_client.invoke(
            FunctionName='fred-ml-processor',
            InvocationType='RequestResponse',
            Payload=json.dumps(test_payload)
        )
        
        response_payload = json.loads(response['Payload'].read().decode('utf-8'))
        
        if response['StatusCode'] == 200 and response_payload.get('status') == 'success':
            print_success("Lambda function executed successfully")
            print_info(f"Report ID: {response_payload.get('report_id')}")
            print_info(f"Report Key: {response_payload.get('report_key')}")
            return response_payload
        else:
            print_error(f"Lambda function failed: {response_payload}")
            return None
            
    except Exception as e:
        print_error(f"Lambda invocation failed: {e}")
        return None

def test_s3_storage():
    """Test S3 storage and retrieval"""
    print_header("Testing S3 Storage")
    
    try:
        s3 = boto3.client('s3', region_name='us-west-2')
        
        # List reports
        response = s3.list_objects_v2(
            Bucket='fredmlv1',
            Prefix='reports/'
        )
        
        if 'Contents' in response:
            print_success(f"Found {len(response['Contents'])} report(s) in S3")
            
            # Get the latest report
            latest_report = max(response['Contents'], key=lambda x: x['LastModified'])
            print_info(f"Latest report: {latest_report['Key']}")
            print_info(f"Size: {latest_report['Size']} bytes")
            print_info(f"Last modified: {latest_report['LastModified']}")
            
            # Download and verify report
            report_response = s3.get_object(
                Bucket='fredmlv1',
                Key=latest_report['Key']
            )
            
            report_data = json.loads(report_response['Body'].read().decode('utf-8'))
            
            # Verify report structure
            required_fields = ['report_id', 'timestamp', 'indicators', 'statistics', 'data']
            for field in required_fields:
                if field not in report_data:
                    print_error(f"Missing required field: {field}")
                    return False
            
            print_success("Report structure is valid")
            print_info(f"Indicators: {report_data['indicators']}")
            print_info(f"Data points: {len(report_data['data'])}")
            
            return latest_report['Key']
        else:
            print_error("No reports found in S3")
            return None
            
    except Exception as e:
        print_error(f"S3 verification failed: {e}")
        return None

def test_visualizations():
    """Test visualization storage"""
    print_header("Testing Visualizations")
    
    try:
        s3 = boto3.client('s3', region_name='us-west-2')
        
        # List visualizations
        response = s3.list_objects_v2(
            Bucket='fredmlv1',
            Prefix='visualizations/'
        )
        
        if 'Contents' in response:
            print_success(f"Found {len(response['Contents'])} visualization(s) in S3")
            
            # Check for specific visualization types
            visualization_types = ['time_series.png', 'correlation.png']
            for viz_type in visualization_types:
                viz_objects = [obj for obj in response['Contents'] if viz_type in obj['Key']]
                if viz_objects:
                    print_success(f"{viz_type}: {len(viz_objects)} file(s)")
                else:
                    print_warning(f"{viz_type}: No files found")
        else:
            print_warning("No visualizations found in S3 (this might be expected)")
        
        return True
        
    except Exception as e:
        print_error(f"Visualization verification failed: {e}")
        return False

def test_streamlit_app():
    """Test Streamlit app components"""
    print_header("Testing Streamlit App")
    
    try:
        # Test configuration loading
        project_root = Path(__file__).parent.parent
        sys.path.append(str(project_root / 'frontend'))
        
        from app import load_config, init_aws_clients
        
        # Test configuration
        config = load_config()
        if config['s3_bucket'] == 'fredmlv1' and config['lambda_function'] == 'fred-ml-processor':
            print_success("Streamlit configuration is correct")
        else:
            print_error("Streamlit configuration mismatch")
            return False
        
        # Test AWS clients
        s3_client, lambda_client = init_aws_clients()
        if s3_client and lambda_client:
            print_success("AWS clients initialized successfully")
        else:
            print_error("Failed to initialize AWS clients")
            return False
        
        return True
        
    except Exception as e:
        print_error(f"Streamlit app test failed: {e}")
        return False

def test_data_quality():
    """Test data quality and completeness"""
    print_header("Testing Data Quality")
    
    try:
        s3 = boto3.client('s3', region_name='us-west-2')
        
        # Get the latest report
        response = s3.list_objects_v2(
            Bucket='fredmlv1',
            Prefix='reports/'
        )
        
        if 'Contents' in response:
            latest_report = max(response['Contents'], key=lambda x: x['LastModified'])
            
            # Download report
            report_response = s3.get_object(
                Bucket='fredmlv1',
                Key=latest_report['Key']
            )
            
            report_data = json.loads(report_response['Body'].read().decode('utf-8'))
            
            # Verify data quality
            if len(report_data['data']) > 0:
                print_success("Data points found")
            else:
                print_error("No data points found")
                return False
            
            if len(report_data['statistics']) > 0:
                print_success("Statistics generated")
            else:
                print_error("No statistics found")
                return False
            
            # Check for requested indicators
            test_indicators = ['GDP', 'UNRATE']
            for indicator in test_indicators:
                if indicator in report_data['indicators']:
                    print_success(f"Indicator '{indicator}' found")
                else:
                    print_error(f"Indicator '{indicator}' missing")
                    return False
            
            # Verify date range
            if report_data['start_date'] == '2024-01-01' and report_data['end_date'] == '2024-01-31':
                print_success("Date range is correct")
            else:
                print_error("Date range mismatch")
                return False
            
            print_success("Data quality verification passed")
            print_info(f"Data points: {len(report_data['data'])}")
            print_info(f"Indicators: {report_data['indicators']}")
            print_info(f"Date range: {report_data['start_date']} to {report_data['end_date']}")
            
            return True
        else:
            print_error("No reports found for data quality verification")
            return False
            
    except Exception as e:
        print_error(f"Data quality verification failed: {e}")
        return False

def test_performance():
    """Test performance metrics"""
    print_header("Testing Performance Metrics")
    
    try:
        cloudwatch = boto3.client('cloudwatch', region_name='us-west-2')
        
        # Get Lambda metrics for the last hour
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=1)
        
        # Get invocation metrics
        response = cloudwatch.get_metric_statistics(
            Namespace='AWS/Lambda',
            MetricName='Invocations',
            Dimensions=[{'Name': 'FunctionName', 'Value': 'fred-ml-processor'}],
            StartTime=start_time,
            EndTime=end_time,
            Period=300,
            Statistics=['Sum']
        )
        
        if response['Datapoints']:
            invocations = sum(point['Sum'] for point in response['Datapoints'])
            print_success(f"Lambda invocations: {invocations}")
        else:
            print_warning("No Lambda invocation metrics found")
        
        # Get duration metrics
        response = cloudwatch.get_metric_statistics(
            Namespace='AWS/Lambda',
            MetricName='Duration',
            Dimensions=[{'Name': 'FunctionName', 'Value': 'fred-ml-processor'}],
            StartTime=start_time,
            EndTime=end_time,
            Period=300,
            Statistics=['Average', 'Maximum']
        )
        
        if response['Datapoints']:
            avg_duration = sum(point['Average'] for point in response['Datapoints']) / len(response['Datapoints'])
            max_duration = max(point['Maximum'] for point in response['Datapoints'])
            print_success(f"Average duration: {avg_duration:.2f}ms")
            print_success(f"Maximum duration: {max_duration:.2f}ms")
        else:
            print_warning("No Lambda duration metrics found")
        
        return True
        
    except Exception as e:
        print_warning(f"Performance metrics test failed: {e}")
        return True  # Don't fail for metrics issues

def generate_test_report(results):
    """Generate test report"""
    print_header("Test Results Summary")
    
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
    """Main test execution"""
    print_header("FRED ML Complete System Test")
    
    # Check prerequisites
    if not check_prerequisites():
        print_error("Prerequisites not met. Exiting.")
        sys.exit(1)
    
    # Run tests
    results = {}
    
    results['AWS Services'] = test_aws_services()
    results['Lambda Function'] = test_lambda_function() is not None
    results['S3 Storage'] = test_s3_storage() is not None
    results['Visualizations'] = test_visualizations()
    results['Streamlit App'] = test_streamlit_app()
    results['Data Quality'] = test_data_quality()
    results['Performance'] = test_performance()
    
    # Generate report
    success = generate_test_report(results)
    
    if success:
        print_header("🎉 All Tests Passed!")
        print_success("FRED ML system is working correctly")
        sys.exit(0)
    else:
        print_header("❌ Some Tests Failed")
        print_error("Please check the detailed report and fix any issues")
        sys.exit(1)

if __name__ == "__main__":
    main() 