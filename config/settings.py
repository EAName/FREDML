"""
Configuration settings for FRED ML application
"""

import os
from typing import Optional

# FRED API Configuration
FRED_API_KEY = os.getenv('FRED_API_KEY', '')

# AWS Configuration
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID', '')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY', '')

# Application Configuration
DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# Performance Configuration
MAX_WORKERS = int(os.getenv('MAX_WORKERS', '10'))  # For parallel processing
REQUEST_TIMEOUT = int(os.getenv('REQUEST_TIMEOUT', '30'))  # API request timeout
CACHE_DURATION = int(os.getenv('CACHE_DURATION', '3600'))  # Cache duration in seconds

# Streamlit Configuration
STREAMLIT_SERVER_PORT = int(os.getenv('STREAMLIT_SERVER_PORT', '8501'))
STREAMLIT_SERVER_ADDRESS = os.getenv('STREAMLIT_SERVER_ADDRESS', '0.0.0.0')

# Data Configuration
DEFAULT_SERIES_LIST = [
    'GDPC1',    # Real GDP
    'INDPRO',   # Industrial Production
    'RSAFS',    # Retail Sales
    'CPIAUCSL', # Consumer Price Index
    'FEDFUNDS', # Federal Funds Rate
    'DGS10',    # 10-Year Treasury
    'UNRATE',   # Unemployment Rate
    'PAYEMS',   # Total Nonfarm Payrolls
    'PCE',      # Personal Consumption Expenditures
    'M2SL',     # M2 Money Stock
    'TCU',      # Capacity Utilization
    'DEXUSEU'   # US/Euro Exchange Rate
]

# Default date ranges
DEFAULT_START_DATE = '2019-01-01'
DEFAULT_END_DATE = '2024-12-31'

# Directory Configuration
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
PLOTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'exports')

# Analysis Configuration
ANALYSIS_TYPES = {
    'comprehensive': 'Comprehensive Analysis',
    'forecasting': 'Time Series Forecasting',
    'segmentation': 'Market Segmentation',
    'statistical': 'Statistical Modeling'
}

class Config:
    @staticmethod
    def get_fred_api_key():
        return FRED_API_KEY

def get_aws_config() -> dict:
    """Get AWS configuration with proper fallbacks"""
    config = {
        'region_name': AWS_REGION,
        'aws_access_key_id': AWS_ACCESS_KEY_ID,
        'aws_secret_access_key': AWS_SECRET_ACCESS_KEY
    }
    
    # Remove empty values to allow boto3 to use default credentials
    config = {k: v for k, v in config.items() if v}
    
    return config

def is_fred_api_configured() -> bool:
    """Check if FRED API is properly configured"""
    return bool(FRED_API_KEY and FRED_API_KEY.strip())

def is_aws_configured() -> bool:
    """Check if AWS is properly configured"""
    return bool(AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY)

def get_analysis_config(analysis_type: str) -> dict:
    """Get configuration for specific analysis type"""
    return {
        'type': analysis_type,
        'name': ANALYSIS_TYPES.get(analysis_type, analysis_type.title()),
        'enabled': True
    } 