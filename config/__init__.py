"""
Configuration package for FRED ML
"""

from .settings import *

__all__ = [
    'FRED_API_KEY',
    'AWS_REGION',
    'AWS_ACCESS_KEY_ID', 
    'AWS_SECRET_ACCESS_KEY',
    'DEBUG',
    'LOG_LEVEL',
    'MAX_WORKERS',
    'REQUEST_TIMEOUT',
    'CACHE_DURATION',
    'STREAMLIT_SERVER_PORT',
    'STREAMLIT_SERVER_ADDRESS',
    'DEFAULT_SERIES_LIST',
    'DEFAULT_START_DATE',
    'DEFAULT_END_DATE',
    'OUTPUT_DIR',
    'PLOTS_DIR',
    'ANALYSIS_TYPES',
    'get_aws_config',
    'is_fred_api_configured',
    'is_aws_configured',
    'get_analysis_config'
] 