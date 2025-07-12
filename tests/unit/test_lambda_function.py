#!/usr/bin/env python3
"""
Unit tests for FRED ML Lambda Function
Tests core functionality without AWS dependencies
"""

import pytest
import sys
import json
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / 'src'))

class TestLambdaFunction:
    """Test cases for Lambda function core functionality"""
    
    @pytest.fixture
    def mock_event(self):
        """Mock Lambda event"""
        return {
            'indicators': ['GDP', 'UNRATE'],
            'start_date': '2024-01-01',
            'end_date': '2024-01-31',
            'options': {
                'visualizations': True,
                'correlation': True,
                'statistics': True
            }
        }
    
    @pytest.fixture
    def mock_context(self):
        """Mock Lambda context"""
        context = Mock()
        context.function_name = 'fred-ml-processor'
        context.function_version = '$LATEST'
        context.invoked_function_arn = 'arn:aws:lambda:us-west-2:123456789012:function:fred-ml-processor'
        context.memory_limit_in_mb = 512
        context.remaining_time_in_millis = 300000
        return context
    
    def test_create_dataframe(self):
        """Test DataFrame creation from series data"""
        from lambda.lambda_function import create_dataframe
        
        # Create mock series data
        dates = pd.date_range('2024-01-01', '2024-01-05', freq='D')
        series_data = {
            'GDP': pd.Series([100.0, 101.0, 102.0, 103.0, 104.0], index=dates),
            'UNRATE': pd.Series([3.5, 3.6, 3.7, 3.8, 3.9], index=dates)
        }
        
        df = create_dataframe(series_data)
        
        assert not df.empty
        assert 'GDP' in df.columns
        assert 'UNRATE' in df.columns
        assert len(df) == 5
        assert df.index.name == 'Date'
    
    def test_generate_statistics(self):
        """Test statistics generation"""
        from lambda.lambda_function import generate_statistics
        
        # Create test DataFrame
        dates = pd.date_range('2024-01-01', '2024-01-05', freq='D')
        df = pd.DataFrame({
            'GDP': [100.0, 101.0, 102.0, 103.0, 104.0],
            'UNRATE': [3.5, 3.6, 3.7, 3.8, 3.9]
        }, index=dates)
        
        stats = generate_statistics(df)
        
        assert 'GDP' in stats
        assert 'UNRATE' in stats
        assert 'mean' in stats['GDP']
        assert 'std' in stats['GDP']
        assert 'min' in stats['GDP']
        assert 'max' in stats['GDP']
        assert 'count' in stats['GDP']
        assert 'missing' in stats['GDP']
        
        # Verify calculations
        assert stats['GDP']['mean'] == 102.0
        assert stats['GDP']['min'] == 100.0
        assert stats['GDP']['max'] == 104.0
        assert stats['GDP']['count'] == 5
    
    def test_create_correlation_matrix(self):
        """Test correlation matrix creation"""
        from lambda.lambda_function import create_correlation_matrix
        
        # Create test DataFrame
        dates = pd.date_range('2024-01-01', '2024-01-05', freq='D')
        df = pd.DataFrame({
            'GDP': [100.0, 101.0, 102.0, 103.0, 104.0],
            'UNRATE': [3.5, 3.6, 3.7, 3.8, 3.9]
        }, index=dates)
        
        corr_matrix = create_correlation_matrix(df)
        
        assert 'GDP' in corr_matrix
        assert 'UNRATE' in corr_matrix
        assert 'GDP' in corr_matrix['GDP']
        assert 'UNRATE' in corr_matrix['UNRATE']
        
        # Verify correlation values
        assert corr_matrix['GDP']['GDP'] == 1.0
        assert corr_matrix['UNRATE']['UNRATE'] == 1.0
    
    @patch('lambda.lambda_function.requests.get')
    def test_get_fred_data_success(self, mock_requests):
        """Test successful FRED data fetching"""
        from lambda.lambda_function import get_fred_data
        
        # Mock successful API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'observations': [
                {'date': '2024-01-01', 'value': '100.0'},
                {'date': '2024-01-02', 'value': '101.0'},
                {'date': '2024-01-03', 'value': '102.0'}
            ]
        }
        mock_requests.return_value = mock_response
        
        # Mock environment variable
        with patch('lambda.lambda_function.FRED_API_KEY', 'test-api-key'):
            result = get_fred_data('GDP', '2024-01-01', '2024-01-03')
        
        assert result is not None
        assert len(result) == 3
        assert result.name == 'GDP'
        assert result.iloc[0] == 100.0
        assert result.iloc[1] == 101.0
        assert result.iloc[2] == 102.0
    
    @patch('lambda.lambda_function.requests.get')
    def test_get_fred_data_failure(self, mock_requests):
        """Test FRED data fetching failure"""
        from lambda.lambda_function import get_fred_data
        
        # Mock failed API response
        mock_response = Mock()
        mock_response.status_code = 404
        mock_requests.return_value = mock_response
        
        result = get_fred_data('INVALID', '2024-01-01', '2024-01-03')
        
        assert result is None
    
    def test_create_dataframe_empty_data(self):
        """Test DataFrame creation with empty data"""
        from lambda.lambda_function import create_dataframe
        
        # Test with empty series data
        df = create_dataframe({})
        assert df.empty
        
        # Test with None values
        df = create_dataframe({'GDP': None, 'UNRATE': None})
        assert df.empty
    
    def test_generate_statistics_empty_data(self):
        """Test statistics generation with empty data"""
        from lambda.lambda_function import generate_statistics
        
        # Test with empty DataFrame
        df = pd.DataFrame()
        stats = generate_statistics(df)
        assert stats == {}
        
        # Test with DataFrame containing only NaN values
        df = pd.DataFrame({
            'GDP': [np.nan, np.nan, np.nan],
            'UNRATE': [np.nan, np.nan, np.nan]
        })
        stats = generate_statistics(df)
        assert 'GDP' in stats
        assert stats['GDP']['count'] == 0
        assert stats['GDP']['missing'] == 3
    
    def test_create_correlation_matrix_empty_data(self):
        """Test correlation matrix creation with empty data"""
        from lambda.lambda_function import create_correlation_matrix
        
        # Test with empty DataFrame
        df = pd.DataFrame()
        corr_matrix = create_correlation_matrix(df)
        assert corr_matrix == {}
        
        # Test with single column
        df = pd.DataFrame({'GDP': [100.0, 101.0, 102.0]})
        corr_matrix = create_correlation_matrix(df)
        assert 'GDP' in corr_matrix
        assert corr_matrix['GDP']['GDP'] == 1.0 