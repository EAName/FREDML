#!/usr/bin/env python3
"""
Unit Tests for Lambda Function
"""

import pytest
import json
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

class TestLambdaFunction:
    """Unit tests for Lambda function"""
    
    @pytest.fixture
    def mock_event(self):
        """Mock event for testing"""
        return {
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
    
    @pytest.fixture
    def mock_context(self):
        """Mock context for testing"""
        context = Mock()
        context.function_name = 'fred-ml-processor'
        context.function_version = '$LATEST'
        context.invoked_function_arn = 'arn:aws:lambda:us-west-2:123456789012:function:fred-ml-processor'
        context.memory_limit_in_mb = 512
        context.remaining_time_in_millis = 300000
        context.log_group_name = '/aws/lambda/fred-ml-processor'
        context.log_stream_name = '2024/01/01/[$LATEST]123456789012'
        return context
    
    @patch('lambda.lambda_function.os.environ.get')
    @patch('lambda.lambda_function.boto3.client')
    def test_lambda_handler_success(self, mock_boto3_client, mock_os_environ, mock_event, mock_context):
        """Test successful Lambda function execution"""
        # Mock environment variables
        mock_os_environ.side_effect = lambda key, default=None: {
            'FRED_API_KEY': 'test-api-key',
            'S3_BUCKET': 'fredmlv1'
        }.get(key, default)
        
        # Mock AWS clients
        mock_s3_client = Mock()
        mock_lambda_client = Mock()
        mock_boto3_client.side_effect = [mock_s3_client, mock_lambda_client]
        
        # Mock FRED API response
        with patch('lambda.lambda_function.requests.get') as mock_requests:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                'observations': [
                    {'date': '2024-01-01', 'value': '100.0'},
                    {'date': '2024-01-02', 'value': '101.0'}
                ]
            }
            mock_requests.return_value = mock_response
            
            # Import and test Lambda function
            sys.path.append(str(project_root / 'lambda'))
            from lambda_function import lambda_handler
            
            response = lambda_handler(mock_event, mock_context)
            
            # Verify response structure
            assert response['statusCode'] == 200
            assert 'body' in response
            
            response_body = json.loads(response['body'])
            assert response_body['status'] == 'success'
            assert 'report_id' in response_body
            assert 'report_key' in response_body
    
    @patch('lambda.lambda_function.os.environ.get')
    def test_lambda_handler_missing_api_key(self, mock_os_environ, mock_event, mock_context):
        """Test Lambda function with missing API key"""
        # Mock missing API key
        mock_os_environ.return_value = None
        
        sys.path.append(str(project_root / 'lambda'))
        from lambda_function import lambda_handler
        
        response = lambda_handler(mock_event, mock_context)
        
        # Should handle missing API key gracefully
        assert response['statusCode'] == 500
        response_body = json.loads(response['body'])
        assert response_body['status'] == 'error'
    
    def test_lambda_handler_invalid_event(self, mock_context):
        """Test Lambda function with invalid event"""
        invalid_event = {}
        
        sys.path.append(str(project_root / 'lambda'))
        from lambda_function import lambda_handler
        
        response = lambda_handler(invalid_event, mock_context)
        
        # Should handle invalid event gracefully
        assert response['statusCode'] == 200 or response['statusCode'] == 500
    
    @patch('lambda.lambda_function.os.environ.get')
    @patch('lambda.lambda_function.boto3.client')
    def test_fred_data_fetching(self, mock_boto3_client, mock_os_environ):
        """Test FRED data fetching functionality"""
        # Mock environment
        mock_os_environ.side_effect = lambda key, default=None: {
            'FRED_API_KEY': 'test-api-key',
            'S3_BUCKET': 'fredmlv1'
        }.get(key, default)
        
        mock_s3_client = Mock()
        mock_lambda_client = Mock()
        mock_boto3_client.side_effect = [mock_s3_client, mock_lambda_client]
        
        sys.path.append(str(project_root / 'lambda'))
        from lambda_function import get_fred_data
        
        # Mock successful API response
        with patch('lambda.lambda_function.requests.get') as mock_requests:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                'observations': [
                    {'date': '2024-01-01', 'value': '100.0'},
                    {'date': '2024-01-02', 'value': '101.0'}
                ]
            }
            mock_requests.return_value = mock_response
            
            result = get_fred_data('GDP', '2024-01-01', '2024-01-31')
            
            assert result is not None
            assert len(result) > 0
    
    @patch('lambda.lambda_function.os.environ.get')
    @patch('lambda.lambda_function.boto3.client')
    def test_dataframe_creation(self, mock_boto3_client, mock_os_environ):
        """Test DataFrame creation from series data"""
        # Mock environment
        mock_os_environ.side_effect = lambda key, default=None: {
            'FRED_API_KEY': 'test-api-key',
            'S3_BUCKET': 'fredmlv1'
        }.get(key, default)
        
        mock_s3_client = Mock()
        mock_lambda_client = Mock()
        mock_boto3_client.side_effect = [mock_s3_client, mock_lambda_client]
        
        from lambda.lambda_function import create_dataframe
        import pandas as pd
        
        # Mock series data
        series_data = {
            'GDP': pd.Series([100.0, 101.0], index=pd.to_datetime(['2024-01-01', '2024-01-02'])),
            'UNRATE': pd.Series([3.5, 3.6], index=pd.to_datetime(['2024-01-01', '2024-01-02']))
        }
        
        df = create_dataframe(series_data)
        
        assert not df.empty
        assert 'GDP' in df.columns
        assert 'UNRATE' in df.columns
        assert len(df) == 2
    
    @patch('lambda.lambda_function.os.environ.get')
    @patch('lambda.lambda_function.boto3.client')
    def test_statistics_generation(self, mock_boto3_client, mock_os_environ):
        """Test statistics generation"""
        # Mock environment
        mock_os_environ.side_effect = lambda key, default=None: {
            'FRED_API_KEY': 'test-api-key',
            'S3_BUCKET': 'fredmlv1'
        }.get(key, default)
        
        mock_s3_client = Mock()
        mock_lambda_client = Mock()
        mock_boto3_client.side_effect = [mock_s3_client, mock_lambda_client]
        
        from lambda.lambda_function import generate_statistics
        import pandas as pd
        
        # Create test DataFrame
        df = pd.DataFrame({
            'GDP': [100.0, 101.0, 102.0],
            'UNRATE': [3.5, 3.6, 3.7]
        })
        
        stats = generate_statistics(df)
        
        assert 'GDP' in stats
        assert 'UNRATE' in stats
        assert 'mean' in stats['GDP']
        assert 'std' in stats['GDP']
        assert 'min' in stats['GDP']
        assert 'max' in stats['GDP']
    
    @patch('lambda.lambda_function.os.environ.get')
    @patch('lambda.lambda_function.boto3.client')
    def test_s3_report_storage(self, mock_boto3_client, mock_os_environ):
        """Test S3 report storage"""
        # Mock environment
        mock_os_environ.side_effect = lambda key, default=None: {
            'FRED_API_KEY': 'test-api-key',
            'S3_BUCKET': 'fredmlv1'
        }.get(key, default)
        
        mock_s3_client = Mock()
        mock_lambda_client = Mock()
        mock_boto3_client.side_effect = [mock_s3_client, mock_lambda_client]
        
        from lambda.lambda_function import save_report_to_s3
        
        # Test report data
        report_data = {
            'report_id': 'test_report_123',
            'timestamp': '2024-01-01T00:00:00',
            'indicators': ['GDP'],
            'data': []
        }
        
        result = save_report_to_s3(report_data, 'fredmlv1', 'test_report_123')
        
        # Verify S3 put_object was called
        mock_s3_client.put_object.assert_called_once()
        call_args = mock_s3_client.put_object.call_args
        assert call_args[1]['Bucket'] == 'fredmlv1'
        assert 'test_report_123' in call_args[1]['Key']
        assert call_args[1]['ContentType'] == 'application/json' 