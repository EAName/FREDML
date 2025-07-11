#!/usr/bin/env python3
"""
FRED ML Development Testing
Simple testing script for development environment
"""

import os
import sys
import json
import time
from pathlib import Path

def test_streamlit_app():
    """Test Streamlit app functionality"""
    print("🎨 Testing Streamlit app...")
    
    try:
        # Test app imports
        sys.path.append('frontend')
        from app import load_config, init_aws_clients
        
        # Test configuration loading
        config = load_config()
        if config:
            print("✅ Streamlit app configuration loaded")
        else:
            print("❌ Failed to load Streamlit app configuration")
            return False
        
        # Test AWS client initialization
        try:
            s3_client, lambda_client = init_aws_clients()
            print("✅ AWS clients initialized")
        except Exception as e:
            print(f"❌ AWS client initialization failed: {str(e)}")
            return False
        
        print("✅ Streamlit app test passed")
        return True
        
    except Exception as e:
        print(f"❌ Streamlit app test failed: {str(e)}")
        return False

def test_lambda_function():
    """Test Lambda function"""
    print("⚡ Testing Lambda function...")
    
    try:
        import boto3
        lambda_client = boto3.client('lambda')
        
        # Get function info
        function_info = lambda_client.get_function(FunctionName='fred-ml-processor')
        print(f"✅ Lambda function found: {function_info['Configuration']['FunctionArn']}")
        
        # Test basic invocation
        test_payload = {
            'indicators': ['GDP', 'UNRATE'],
            'start_date': '2023-01-01',
            'end_date': '2023-12-31',
            'test_mode': True
        }
        
        response = lambda_client.invoke(
            FunctionName='fred-ml-processor',
            InvocationType='RequestResponse',
            Payload=json.dumps(test_payload)
        )
        
        if response['StatusCode'] == 200:
            print("✅ Lambda function invocation successful")
            return True
        else:
            print(f"❌ Lambda invocation failed with status {response['StatusCode']}")
            return False
            
    except Exception as e:
        print(f"❌ Lambda function test failed: {str(e)}")
        return False

def test_s3_access():
    """Test S3 bucket access"""
    print("📦 Testing S3 bucket access...")
    
    try:
        import boto3
        s3 = boto3.client('s3')
        
        # Test bucket access
        s3.head_bucket(Bucket='fredmlv1')
        print("✅ S3 bucket access successful")
        
        # Test upload/download
        test_data = "test content"
        test_key = f"dev-test/test-{int(time.time())}.txt"
        
        # Upload test file
        s3.put_object(
            Bucket='fredmlv1',
            Key=test_key,
            Body=test_data.encode('utf-8')
        )
        print("✅ S3 upload successful")
        
        # Download and verify
        response = s3.get_object(Bucket='fredmlv1', Key=test_key)
        downloaded_data = response['Body'].read().decode('utf-8')
        
        if downloaded_data == test_data:
            print("✅ S3 download successful")
        else:
            print("❌ S3 download data mismatch")
            return False
        
        # Clean up test file
        s3.delete_object(Bucket='fredmlv1', Key=test_key)
        print("✅ S3 cleanup successful")
        
        return True
        
    except Exception as e:
        print(f"❌ S3 access test failed: {str(e)}")
        return False

def test_fred_api():
    """Test FRED API access"""
    print("📊 Testing FRED API...")
    
    try:
        from fredapi import Fred
        fred = Fred(api_key=os.getenv('FRED_API_KEY'))
        
        # Test basic API access
        test_series = fred.get_series('GDP', limit=5)
        if len(test_series) > 0:
            print(f"✅ FRED API access successful - retrieved {len(test_series)} data points")
            return True
        else:
            print("❌ FRED API returned no data")
            return False
            
    except Exception as e:
        print(f"❌ FRED API test failed: {str(e)}")
        return False

def test_data_processing():
    """Test data processing capabilities"""
    print("📈 Testing data processing...")
    
    try:
        import pandas as pd
        import numpy as np
        from fredapi import Fred
        
        fred = Fred(api_key=os.getenv('FRED_API_KEY'))
        
        # Get test data
        test_data = {}
        indicators = ['GDP', 'UNRATE', 'CPIAUCSL']
        
        for indicator in indicators:
            try:
                data = fred.get_series(indicator, limit=100)
                test_data[indicator] = data
                print(f"✅ Retrieved {indicator}: {len(data)} observations")
            except Exception as e:
                print(f"❌ Failed to retrieve {indicator}: {str(e)}")
        
        if not test_data:
            print("❌ No test data retrieved")
            return False
        
        # Test data processing
        df = pd.DataFrame(test_data)
        df = df.dropna()
        
        if len(df) > 0:
            # Test basic statistics
            summary = df.describe()
            correlation = df.corr()
            
            print(f"✅ Data processing successful - {len(df)} data points processed")
            print(f"   Summary statistics calculated")
            print(f"   Correlation matrix shape: {correlation.shape}")
            return True
        else:
            print("❌ No valid data after processing")
            return False
            
    except Exception as e:
        print(f"❌ Data processing test failed: {str(e)}")
        return False

def test_visualization():
    """Test visualization generation"""
    print("🎨 Testing visualization generation...")
    
    try:
        import matplotlib.pyplot as plt
        import plotly.express as px
        import seaborn as sns
        import pandas as pd
        import numpy as np
        
        # Create test data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', '2024-01-01', freq='M')
        test_data = pd.DataFrame({
            'GDP': np.random.normal(100, 5, len(dates)),
            'UNRATE': np.random.normal(5, 1, len(dates)),
            'CPIAUCSL': np.random.normal(200, 10, len(dates))
        }, index=dates)
        
        # Test matplotlib
        fig, ax = plt.subplots(figsize=(10, 6))
        test_data.plot(ax=ax)
        plt.title('Test Visualization')
        plt.close()  # Don't display, just test creation
        print("✅ Matplotlib visualization created")
        
        # Test plotly
        fig = px.line(test_data, title='Test Plotly Visualization')
        fig.update_layout(showlegend=True)
        print("✅ Plotly visualization created")
        
        # Test seaborn
        plt.figure(figsize=(8, 6))
        sns.heatmap(test_data.corr(), annot=True, cmap='coolwarm')
        plt.title('Test Correlation Heatmap')
        plt.close()
        print("✅ Seaborn visualization created")
        
        print("✅ All visualization tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Visualization test failed: {str(e)}")
        return False

def main():
    """Main testing function"""
    print("🧪 FRED ML Development Testing")
    print("=" * 50)
    
    tests = [
        ("Streamlit App", test_streamlit_app),
        ("Lambda Function", test_lambda_function),
        ("S3 Bucket Access", test_s3_access),
        ("FRED API", test_fred_api),
        ("Data Processing", test_data_processing),
        ("Visualization", test_visualization)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name} test...")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} test failed")
    
    print(f"\n📊 Test Summary: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All development tests passed!")
        print("\n🎯 Your development environment is ready!")
        print("You can now:")
        print("1. Run the Streamlit app: streamlit run frontend/app.py")
        print("2. Test the complete system: python scripts/test_complete_system.py")
        return True
    else:
        print("❌ Some tests failed. Please check the issues above.")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1) 