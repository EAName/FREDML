"""
FRED Economic Data Analysis Package

A comprehensive tool for collecting, analyzing, and visualizing 
Federal Reserve Economic Data (FRED) using the FRED API.

Author: Economic Data Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Economic Data Team"
__email__ = "data-team@company.com"

from .analysis.advanced_analytics import AdvancedAnalytics
from .core.fred_client import FREDDataCollectorV2

__all__ = [
    "FREDDataCollectorV2",
    "AdvancedAnalytics",
]
