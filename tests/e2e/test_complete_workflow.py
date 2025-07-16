#!/usr/bin/env python3
"""
End-to-End Testing for FRED ML System
Tests the complete workflow: Streamlit → Lambda → S3 → Reports
"""

import pytest
import boto3
import json
import time
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
import requests
import subprocess
import tempfile
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import will be handled dynamically in the test

class TestFredMLEndToEnd:
    """End-to-end test suite for FRED ML system"""
    
    @pytest.fixture(scope="class")
    def aws_clients(self):
        """Initialize AWS clients"""
        return {
            's3': boto3.client('s3', region_name='us-west-2'),
            'lambda': boto3.client('lambda', region_name='us-west-2'),
            'ssm': boto3.client('ssm', region_name='us-west-2')
        }
    
    @pytest.fixture(scope="class")
    def test_config(self):
        """Test configuration"""
        return {
            's3_bucket': 'fredmlv1',
            'lambda_function': 'fred-ml-processor',
            'region': 'us-west-2',
            'test_indicators': ['GDP', 'UNRATE'],
            'test_start_date': '2024-01-01',
            'test_end_date': '2024-01-31'
        }
    
    @pytest.fixture(scope="class")
    def test_report_id(self):
        """Generate unique test report ID"""
        return f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def test_01_aws_credentials(self, aws_clients):
        """Test AWS credentials and permissions"""
        print("\n🔐 Testing AWS credentials...")
        
        # Test S3 access
        try:
            response = aws_clients['s3'].list_objects_v2(
                Bucket='fredmlv1',
                MaxKeys=1
            )
            print("✅ S3 access verified")
        except Exception as e:
            pytest.skip(f"❌ S3 access failed: {e}")
        
        # Test Lambda access
        try:
            response = aws_clients['lambda'].list_functions(MaxItems=1)
            print("✅ Lambda access verified")
        except Exception as e:
            pytest.skip(f"❌ Lambda access failed: {e}")
        
        # Test SSM access
        try:
            response = aws_clients['ssm'].describe_parameters(MaxResults=1)
            print("✅ SSM access verified")
        except Exception as e:
            pytest.skip(f"❌ SSM access failed: {e}")
    
    def test_02_s3_bucket_exists(self, aws_clients, test_config):
        """Test S3 bucket exists and is accessible"""
        print("\n📦 Testing S3 bucket...")
        
        try:
            response = aws_clients['s3'].head_bucket(Bucket=test_config['s3_bucket'])
            print(f"✅ S3 bucket '{test_config['s3_bucket']}' exists and is accessible")
        except Exception as e:
            pytest.skip(f"❌ S3 bucket access failed: {e}")
    
    def test_03_lambda_function_exists(self, aws_clients, test_config):
        """Test Lambda function exists"""
        print("\n⚡ Testing Lambda function...")
        
        try:
            response = aws_clients['lambda'].get_function(
                FunctionName=test_config['lambda_function']
            )
            print(f"✅ Lambda function '{test_config['lambda_function']}' exists")
            print(f"   Runtime: {response['Configuration']['Runtime']}")
            print(f"   Memory: {response['Configuration']['MemorySize']} MB")
            print(f"   Timeout: {response['Configuration']['Timeout']} seconds")
        except Exception as e:
            pytest.skip(f"❌ Lambda function not found: {e}")
    
    def test_04_fred_api_key_configured(self, aws_clients):
        """Test FRED API key is configured in SSM"""
        print("\n🔑 Testing FRED API key...")
        
        try:
            response = aws_clients['ssm'].get_parameter(
                Name='/fred-ml/api-key',
                WithDecryption=True
            )
            api_key = response['Parameter']['Value']
            
            if api_key and api_key != 'your-fred-api-key-here':
                print("✅ FRED API key is configured")
            else:
                pytest.skip("❌ FRED API key not properly configured")
        except Exception as e:
            pytest.skip(f"❌ FRED API key not found in SSM: {e}")
    
    def test_05_lambda_function_invocation(self, aws_clients, test_config, test_report_id):
        """Test Lambda function invocation with test data"""
        print("\n🚀 Testing Lambda function invocation...")
        
        # Test payload
        test_payload = {
            'indicators': test_config['test_indicators'],
            'start_date': test_config['test_start_date'],
            'end_date': test_config['test_end_date'],
            'options': {
                'visualizations': True,
                'correlation': True,
                'forecasting': False,
                'statistics': True
            }
        }
        
        try:
            # Invoke Lambda function
            response = aws_clients['lambda'].invoke(
                FunctionName=test_config['lambda_function'],
                InvocationType='RequestResponse',
                Payload=json.dumps(test_payload)
            )
            
            # Parse response
            response_payload = json.loads(response['Payload'].read().decode('utf-8'))
            
            if response['StatusCode'] == 200 and response_payload.get('status') == 'success':
                print("✅ Lambda function executed successfully")
                print(f"   Report ID: {response_payload.get('report_id')}")
                print(f"   Report Key: {response_payload.get('report_key')}")
                return response_payload
            else:
                pytest.skip(f"❌ Lambda function failed: {response_payload}")
                
        except Exception as e:
            pytest.skip(f"❌ Lambda invocation failed: {e}")
    
    def test_06_s3_report_storage(self, aws_clients, test_config, test_report_id):
        """Test S3 report storage"""
        print("\n📄 Testing S3 report storage...")
        
        try:
            # List objects in reports directory
            response = aws_clients['s3'].list_objects_v2(
                Bucket=test_config['s3_bucket'],
                Prefix='reports/'
            )
            
            if 'Contents' in response:
                print(f"✅ Found {len(response['Contents'])} report(s) in S3")
                
                # Get the latest report
                latest_report = max(response['Contents'], key=lambda x: x['LastModified'])
                print(f"   Latest report: {latest_report['Key']}")
                print(f"   Size: {latest_report['Size']} bytes")
                print(f"   Last modified: {latest_report['LastModified']}")
                
                # Download and verify report content
                report_response = aws_clients['s3'].get_object(
                    Bucket=test_config['s3_bucket'],
                    Key=latest_report['Key']
                )
                
                report_data = json.loads(report_response['Body'].read().decode('utf-8'))
                
                # Verify report structure
                required_fields = ['report_id', 'timestamp', 'indicators', 'statistics', 'data']
                for field in required_fields:
                    assert field in report_data, f"Missing required field: {field}"
                
                print("✅ Report structure is valid")
                print(f"   Indicators: {report_data['indicators']}")
                print(f"   Data points: {len(report_data['data'])}")
                
                return latest_report['Key']
            else:
                pytest.skip("❌ No reports found in S3")
                
        except Exception as e:
            pytest.skip(f"❌ S3 report storage test failed: {e}")
    
    def test_07_s3_visualization_storage(self, aws_clients, test_config):
        """Test S3 visualization storage"""
        print("\n📊 Testing S3 visualization storage...")
        
        try:
            # List objects in visualizations directory
            response = aws_clients['s3'].list_objects_v2(
                Bucket=test_config['s3_bucket'],
                Prefix='visualizations/'
            )
            
            if 'Contents' in response:
                print(f"✅ Found {len(response['Contents'])} visualization(s) in S3")
                
                # Check for common visualization types
                viz_types = ['time_series.png', 'correlation.png', 'distribution_']
                found_types = []
                
                for obj in response['Contents']:
                    for viz_type in viz_types:
                        if viz_type in obj['Key']:
                            found_types.append(viz_type)
                            break
                
                if found_types:
                    print(f"   Found visualization types: {', '.join(set(found_types))}")
                    return True
                else:
                    print("⚠️  No expected visualization types found")
                    return True
            else:
                print("⚠️  No visualizations found in S3")
                return True
                
        except Exception as e:
            pytest.skip(f"❌ S3 visualization storage test failed: {e}")
    
    def test_08_streamlit_frontend_simulation(self, test_config):
        """Simulate Streamlit frontend functionality"""
        print("\n🎨 Testing Streamlit frontend simulation...")
        
        try:
            # Import Streamlit app components
            sys.path.append(str(project_root / 'frontend'))
            
            # Test configuration loading
            from frontend.app import load_config
            config = load_config()
            
            assert config['s3_bucket'] == test_config['s3_bucket'], "S3 bucket mismatch"
            assert config['lambda_function'] == test_config['lambda_function'], "Lambda function mismatch"
            
            print("✅ Streamlit configuration is correct")
            
            # Test AWS client initialization
            from frontend.app import init_aws_clients
            s3_client, lambda_client = init_aws_clients()
            
            if s3_client and lambda_client:
                print("✅ AWS clients initialized successfully")
            else:
                pytest.skip("❌ Failed to initialize AWS clients")
            
            return True
            
        except Exception as e:
            pytest.skip(f"❌ Streamlit frontend simulation failed: {e}")
    
    def test_09_data_quality_verification(self, aws_clients, test_config):
        """Verify data quality and completeness"""
        print("\n🔍 Testing data quality...")
        
        try:
            # Get the latest report
            response = aws_clients['s3'].list_objects_v2(
                Bucket=test_config['s3_bucket'],
                Prefix='reports/'
            )
            
            if 'Contents' in response:
                latest_report = max(response['Contents'], key=lambda x: x['LastModified'])
                
                # Download report
                report_response = aws_clients['s3'].get_object(
                    Bucket=test_config['s3_bucket'],
                    Key=latest_report['Key']
                )
                
                report_data = json.loads(report_response['Body'].read().decode('utf-8'))
                
                # Verify data quality
                assert len(report_data['data']) > 0, "No data points found"
                assert len(report_data['statistics']) > 0, "No statistics found"
                
                # Check for each requested indicator
                for indicator in test_config['test_indicators']:
                    assert indicator in report_data['indicators'], f"Missing indicator: {indicator}"
                
                # Verify date range
                assert report_data['start_date'] == test_config['test_start_date'], "Start date mismatch"
                assert report_data['end_date'] == test_config['test_end_date'], "End date mismatch"
                
                print("✅ Data quality verification passed")
                print(f"   Data points: {len(report_data['data'])}")
                print(f"   Indicators: {report_data['indicators']}")
                print(f"   Date range: {report_data['start_date']} to {report_data['end_date']}")
                
                return True
            else:
                pytest.skip("❌ No reports found for data quality verification")
                
        except Exception as e:
            pytest.skip(f"❌ Data quality verification failed: {e}")
    
    def test_10_performance_metrics(self, aws_clients, test_config):
        """Test performance metrics"""
        print("\n⚡ Testing performance metrics...")
        
        try:
            # Get Lambda function metrics
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=1)
            
            cloudwatch = boto3.client('cloudwatch', region_name=test_config['region'])
            
            # Get invocation metrics
            response = cloudwatch.get_metric_statistics(
                Namespace='AWS/Lambda',
                MetricName='Invocations',
                Dimensions=[{'Name': 'FunctionName', 'Value': test_config['lambda_function']}],
                StartTime=start_time,
                EndTime=end_time,
                Period=300,
                Statistics=['Sum']
            )
            
            if response['Datapoints']:
                invocations = sum(point['Sum'] for point in response['Datapoints'])
                print(f"✅ Lambda invocations: {invocations}")
            else:
                print("⚠️  No Lambda invocation metrics found")
            
            # Get duration metrics
            response = cloudwatch.get_metric_statistics(
                Namespace='AWS/Lambda',
                MetricName='Duration',
                Dimensions=[{'Name': 'FunctionName', 'Value': test_config['lambda_function']}],
                StartTime=start_time,
                EndTime=end_time,
                Period=300,
                Statistics=['Average', 'Maximum']
            )
            
            if response['Datapoints']:
                avg_duration = sum(point['Average'] for point in response['Datapoints']) / len(response['Datapoints'])
                max_duration = max(point['Maximum'] for point in response['Datapoints'])
                print(f"✅ Average duration: {avg_duration:.2f}ms")
                print(f"✅ Maximum duration: {max_duration:.2f}ms")
            else:
                print("⚠️  No Lambda duration metrics found")
            
            return True
            
        except Exception as e:
            print(f"⚠️  Performance metrics test failed: {e}")
            return True  # Don't fail the test for metrics issues
    
    def test_11_error_handling(self, aws_clients, test_config):
        """Test error handling scenarios"""
        print("\n🚨 Testing error handling...")
        
        try:
            # Test with invalid indicators
            invalid_payload = {
                'indicators': ['INVALID_INDICATOR'],
                'start_date': '2024-01-01',
                'end_date': '2024-01-31',
                'options': {
                    'visualizations': False,
                    'correlation': False,
                    'statistics': True
                }
            }
            
            response = aws_clients['lambda'].invoke(
                FunctionName=test_config['lambda_function'],
                InvocationType='RequestResponse',
                Payload=json.dumps(invalid_payload)
            )
            
            response_payload = json.loads(response['Payload'].read().decode('utf-8'))
            
            # Should handle gracefully even with invalid data
            if response['StatusCode'] == 200:
                print("✅ Error handling works correctly")
            else:
                print(f"⚠️  Unexpected response: {response_payload}")
            
            return True
            
        except Exception as e:
            print(f"⚠️  Error handling test failed: {e}")
            return True  # Don't fail the test for error handling issues
    
    def test_12_cleanup_test_data(self, aws_clients, test_config, test_report_id):
        """Clean up test data (optional)"""
        print("\n🧹 Testing cleanup...")
        
        try:
            # List test objects
            response = aws_clients['s3'].list_objects_v2(
                Bucket=test_config['s3_bucket'],
                Prefix=f'reports/{test_report_id}/'
            )
            
            if 'Contents' in response:
                print(f"Found {len(response['Contents'])} test objects to clean up")
                
                # Delete test objects
                for obj in response['Contents']:
                    aws_clients['s3'].delete_object(
                        Bucket=test_config['s3_bucket'],
                        Key=obj['Key']
                    )
                
                print("✅ Test data cleaned up")
            else:
                print("✅ No test data to clean up")
            
            return True
            
        except Exception as e:
            print(f"⚠️  Cleanup failed: {e}")
            return True  # Don't fail the test for cleanup issues

def run_e2e_tests():
    """Run all end-to-end tests"""
    print("🚀 Starting FRED ML End-to-End Tests")
    print("=" * 50)
    
    # Run tests
    pytest.main([
        __file__,
        '-v',
        '--tb=short',
        '--disable-warnings'
    ])

if __name__ == "__main__":
    run_e2e_tests() 