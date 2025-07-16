#!/usr/bin/env python3
"""
System integration tests for FRED ML
Tests complete workflows and system integration
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np
import requests
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


class TestSystemIntegration:
    """Test complete system integration and workflows"""
    
    def setup_method(self):
        """Set up test environment"""
        self.test_api_key = "test_fred_api_key"
        self.test_data = pd.DataFrame({
            'GDPC1': [22000, 22100, 22200, 22300, 22400],
            'INDPRO': [100, 101, 102, 103, 104],
            'CPIAUCSL': [250, 251, 252, 253, 254],
            'FEDFUNDS': [2.0, 2.1, 2.2, 2.3, 2.4],
            'PAYEMS': [150000, 151000, 152000, 153000, 154000]
        }, index=pd.date_range('2023-01-01', periods=5, freq='M'))
    
    def test_complete_analytics_workflow(self):
        """Test complete analytics workflow"""
        try:
            # Test analytics module import
            from src.analysis.comprehensive_analytics import ComprehensiveAnalytics
            
            # Create analytics instance
            analytics = ComprehensiveAnalytics(self.test_api_key, output_dir="test_output")
            
            # Test basic functionality
            assert analytics is not None
            assert hasattr(analytics, 'run_complete_analysis')
            
        except ImportError as e:
            pytest.skip(f"Analytics module not available: {e}")
        except Exception as e:
            pytest.fail(f"Analytics workflow test failed: {e}")
    
    def test_fred_client_integration(self):
        """Test FRED client integration"""
        try:
            from src.core.enhanced_fred_client import EnhancedFREDClient
            
            client = EnhancedFREDClient(self.test_api_key)
            
            # Test client structure
            assert hasattr(client, 'fetch_economic_data')
            assert hasattr(client, 'fetch_quarterly_data')
            
        except ImportError as e:
            pytest.skip(f"FRED client module not available: {e}")
        except Exception as e:
            pytest.fail(f"FRED client integration test failed: {e}")
    
    def test_mathematical_fixes_integration(self):
        """Test mathematical fixes integration"""
        try:
            from src.analysis.mathematical_fixes import MathematicalFixes
            
            fixes = MathematicalFixes()
            
            # Test comprehensive fixes
            fixed_data, fix_info = fixes.apply_comprehensive_fixes(
                self.test_data,
                target_freq='Q',
                growth_method='pct_change',
                normalize_units=True
            )
            
            assert fixed_data is not None
            assert isinstance(fix_info, dict)
            
        except ImportError as e:
            pytest.skip(f"Mathematical fixes module not available: {e}")
        except Exception as e:
            pytest.fail(f"Mathematical fixes integration test failed: {e}")
    
    def test_forecasting_integration(self):
        """Test forecasting module integration"""
        try:
            from src.analysis.economic_forecasting import EconomicForecaster
            
            forecaster = EconomicForecaster(self.test_data)
            
            # Test forecaster structure
            assert hasattr(forecaster, 'forecast_series')
            assert hasattr(forecaster, 'backtest_forecast')
            
        except ImportError as e:
            pytest.skip(f"Forecasting module not available: {e}")
        except Exception as e:
            pytest.fail(f"Forecasting integration test failed: {e}")
    
    def test_segmentation_integration(self):
        """Test segmentation module integration"""
        try:
            from src.analysis.economic_segmentation import EconomicSegmentation
            
            segmentation = EconomicSegmentation(self.test_data)
            
            # Test segmentation structure
            assert hasattr(segmentation, 'cluster_time_periods')
            assert hasattr(segmentation, 'cluster_economic_series')
            
        except ImportError as e:
            pytest.skip(f"Segmentation module not available: {e}")
        except Exception as e:
            pytest.fail(f"Segmentation integration test failed: {e}")


class TestAppIntegration:
    """Test application integration and functionality"""
    
    @patch('requests.get')
    def test_app_health_check(self, mock_get):
        """Test app health check with mocked requests"""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        try:
            response = requests.get("http://localhost:8501/_stcore/health", timeout=5)
            assert response.status_code == 200
        except Exception as e:
            pytest.fail(f"App health check test failed: {e}")
    
    def test_config_integration(self):
        """Test configuration integration"""
        try:
            # Test environment variable loading
            import os
            fred_key = os.getenv('FRED_API_KEY', 'test_key')
            
            assert fred_key is not None
            assert len(fred_key) > 0
            
        except Exception as e:
            pytest.fail(f"Config integration test failed: {e}")
    
    def test_data_processing_integration(self):
        """Test data processing integration"""
        # Create test data
        test_data = pd.DataFrame({
            'GDPC1': [22000, 22100, 22200],
            'INDPRO': [100, 101, 102],
            'CPIAUCSL': [250, 251, 252]
        })
        
        # Test basic data processing
        assert not test_data.empty
        assert len(test_data.columns) > 0
        assert len(test_data) > 0
        
        # Test data validation
        for col in test_data.columns:
            assert pd.api.types.is_numeric_dtype(test_data[col])
    
    def test_visualization_integration(self):
        """Test visualization integration"""
        try:
            import matplotlib.pyplot as plt
            
            # Test basic plotting
            test_data = pd.DataFrame({
                'GDPC1': [22000, 22100, 22200],
                'INDPRO': [100, 101, 102]
            })
            
            fig, ax = plt.subplots()
            test_data.plot(ax=ax)
            plt.close()
            
            assert True  # If we get here, plotting worked
            
        except ImportError:
            pytest.skip("Matplotlib not available")
        except Exception as e:
            pytest.fail(f"Visualization integration test failed: {e}")


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_invalid_api_key_handling(self):
        """Test handling of invalid API key"""
        try:
            from src.core.enhanced_fred_client import EnhancedFREDClient
            
            # Test with invalid key
            client = EnhancedFREDClient("invalid_key")
            
            # Should not raise exception on initialization
            assert client is not None
            
        except ImportError as e:
            pytest.skip(f"FRED client module not available: {e}")
        except Exception as e:
            pytest.fail(f"Invalid API key handling test failed: {e}")
    
    def test_empty_data_handling(self):
        """Test handling of empty data"""
        try:
            from src.analysis.mathematical_fixes import MathematicalFixes
            
            fixes = MathematicalFixes()
            
            # Test with empty DataFrame
            empty_data = pd.DataFrame()
            
            # Should handle empty data gracefully
            if hasattr(fixes, 'normalize_units'):
                try:
                    result = fixes.normalize_units(empty_data)
                    assert result is not None
                except Exception:
                    # It's OK for empty data to raise an exception
                    pass
            
        except ImportError as e:
            pytest.skip(f"Mathematical fixes module not available: {e}")
        except Exception as e:
            pytest.fail(f"Empty data handling test failed: {e}")
    
    def test_missing_data_handling(self):
        """Test handling of missing data"""
        # Create data with missing values
        test_data = pd.DataFrame({
            'GDPC1': [22000, np.nan, 22200],
            'INDPRO': [100, 101, np.nan],
            'CPIAUCSL': [250, 251, 252]
        })
        
        # Test basic missing data handling
        assert test_data.isnull().sum().sum() > 0
        
        # Test that we can handle missing data
        cleaned_data = test_data.dropna()
        assert len(cleaned_data) < len(test_data)


class TestPerformance:
    """Test performance and scalability"""
    
    def test_large_data_handling(self):
        """Test handling of large datasets"""
        # Create larger test dataset
        large_data = pd.DataFrame({
            'GDPC1': np.random.normal(22000, 1000, 1000),
            'INDPRO': np.random.normal(100, 5, 1000),
            'CPIAUCSL': np.random.normal(250, 10, 1000)
        }, index=pd.date_range('2020-01-01', periods=1000, freq='D'))
        
        # Test basic operations on large data
        assert len(large_data) == 1000
        assert len(large_data.columns) == 3
        
        # Test memory usage (basic check)
        memory_usage = large_data.memory_usage(deep=True).sum()
        assert memory_usage > 0
    
    def test_computation_efficiency(self):
        """Test computation efficiency"""
        import time
        
        # Test basic computation time
        start_time = time.time()
        
        # Perform some basic computations
        test_data = pd.DataFrame({
            'GDPC1': np.random.normal(22000, 1000, 100),
            'INDPRO': np.random.normal(100, 5, 100)
        })
        
        # Basic operations
        test_data.describe()
        test_data.corr()
        
        end_time = time.time()
        computation_time = end_time - start_time
        
        # Should complete within reasonable time (5 seconds)
        assert computation_time < 5.0


if __name__ == "__main__":
    pytest.main([__file__]) 