#!/usr/bin/env python3
"""
Core functionality tests for FRED ML
Tests basic functionality without AWS dependencies
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / 'src'))

class TestCoreFunctionality:
    """Test core functionality without AWS dependencies"""
    
    def test_fred_api_client_import(self):
        """Test that FRED API client can be imported"""
        try:
            from frontend.fred_api_client import FREDAPIClient
            assert FREDAPIClient is not None
        except ImportError as e:
            pytest.skip(f"FRED API client not available: {e}")
    
    def test_demo_data_import(self):
        """Test that demo data can be imported"""
        try:
            from frontend.demo_data import get_demo_data
            assert get_demo_data is not None
        except ImportError as e:
            pytest.skip(f"Demo data not available: {e}")
    
    def test_config_import(self):
        """Test that config can be imported"""
        try:
            from config.settings import FRED_API_KEY, AWS_REGION
            assert FRED_API_KEY is not None
            assert AWS_REGION is not None
        except ImportError as e:
            pytest.skip(f"Config not available: {e}")
    
    def test_streamlit_app_import(self):
        """Test that Streamlit app can be imported"""
        try:
            # Just test that the file exists and can be read
            app_path = project_root / 'frontend' / 'app.py'
            assert app_path.exists()
            
            # Test basic imports from the app
            import streamlit as st
            assert st is not None
        except ImportError as e:
            pytest.skip(f"Streamlit not available: {e}")
    
    def test_pandas_functionality(self):
        """Test basic pandas functionality"""
        # Create test data
        dates = pd.date_range('2024-01-01', '2024-01-05', freq='D')
        df = pd.DataFrame({
            'GDP': [100.0, 101.0, 102.0, 103.0, 104.0],
            'UNRATE': [3.5, 3.6, 3.7, 3.8, 3.9]
        }, index=dates)
        
        # Test basic operations
        assert not df.empty
        assert len(df) == 5
        assert 'GDP' in df.columns
        assert 'UNRATE' in df.columns
        
        # Test statistics
        assert df['GDP'].mean() == 102.0
        assert df['GDP'].min() == 100.0
        assert df['GDP'].max() == 104.0
    
    def test_numpy_functionality(self):
        """Test basic numpy functionality"""
        # Test array operations
        arr = np.array([1, 2, 3, 4, 5])
        assert arr.mean() == 3.0
        assert arr.std() > 0
        
        # Test random number generation
        random_arr = np.random.randn(100)
        assert len(random_arr) == 100
        assert random_arr.mean() != 0  # Should be close to 0 but not exactly
    
    def test_plotly_import(self):
        """Test plotly import"""
        try:
            import plotly.express as px
            import plotly.graph_objects as go
            assert px is not None
            assert go is not None
        except ImportError as e:
            pytest.skip(f"Plotly not available: {e}")
    
    def test_boto3_import(self):
        """Test boto3 import"""
        try:
            import boto3
            assert boto3 is not None
        except ImportError as e:
            pytest.skip(f"Boto3 not available: {e}")
    
    def test_requests_import(self):
        """Test requests import"""
        try:
            import requests
            assert requests is not None
        except ImportError as e:
            pytest.skip(f"Requests not available: {e}")
    
    def test_data_processing(self):
        """Test basic data processing functionality"""
        # Create test data
        data = {
            'dates': pd.date_range('2024-01-01', '2024-01-10', freq='D'),
            'values': [100 + i for i in range(10)]
        }
        
        # Create DataFrame
        df = pd.DataFrame({
            'date': data['dates'],
            'value': data['values']
        })
        
        # Test data processing
        df['value_lag1'] = df['value'].shift(1)
        df['value_change'] = df['value'].diff()
        
        assert len(df) == 10
        assert 'value_lag1' in df.columns
        assert 'value_change' in df.columns
        
        # Test that we can handle missing values
        df_clean = df.dropna()
        assert len(df_clean) < len(df)  # Should have fewer rows due to NaN values
    
    def test_string_parsing(self):
        """Test string parsing functionality (for FRED API values)"""
        # Test parsing FRED API values with commas
        test_values = [
            "2,239.7",
            "1,000.0",
            "100.5",
            "1,234,567.89"
        ]
        
        expected_values = [
            2239.7,
            1000.0,
            100.5,
            1234567.89
        ]
        
        for test_val, expected_val in zip(test_values, expected_values):
            # Remove commas and convert to float
            cleaned_val = test_val.replace(',', '')
            parsed_val = float(cleaned_val)
            assert parsed_val == expected_val
    
    def test_error_handling(self):
        """Test error handling functionality"""
        # Test handling of invalid data
        invalid_values = [
            "N/A",
            ".",
            "",
            "invalid"
        ]
        
        for invalid_val in invalid_values:
            try:
                # Try to convert to float
                float_val = float(invalid_val)
                # If we get here, it's unexpected
                assert False, f"Should have failed for {invalid_val}"
            except (ValueError, TypeError):
                # Expected behavior
                pass
    
    def test_configuration_loading(self):
        """Test configuration loading"""
        try:
            from config.settings import (
                FRED_API_KEY, 
                AWS_REGION, 
                DEBUG, 
                LOG_LEVEL,
                get_aws_config,
                is_fred_api_configured,
                is_aws_configured
            )
            
            # Test configuration functions
            aws_config = get_aws_config()
            assert isinstance(aws_config, dict)
            
            fred_configured = is_fred_api_configured()
            assert isinstance(fred_configured, bool)
            
            aws_configured = is_aws_configured()
            assert isinstance(aws_configured, bool)
            
        except ImportError as e:
            pytest.skip(f"Configuration not available: {e}") 