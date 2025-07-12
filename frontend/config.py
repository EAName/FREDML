"""
FRED ML - Configuration Settings
Configuration for FRED API and application settings
"""

import os
from typing import Optional

class Config:
    """Configuration class for FRED ML application"""
    
    # FRED API Configuration
    FRED_API_KEY: Optional[str] = os.getenv('FRED_API_KEY')
    
    # Application Settings
    APP_TITLE = "FRED ML - Economic Analytics Platform"
    APP_DESCRIPTION = "Enterprise-grade economic analytics and forecasting platform"
    
    # Data Settings
    DEFAULT_START_DATE = "2020-01-01"
    DEFAULT_END_DATE = "2024-12-31"
    
    # Analysis Settings
    FORECAST_PERIODS = 12
    CONFIDENCE_LEVEL = 0.95
    
    # UI Settings
    THEME_COLOR = "#1f77b4"
    SUCCESS_COLOR = "#2ca02c"
    WARNING_COLOR = "#ff7f0e"
    ERROR_COLOR = "#d62728"
    
    @classmethod
    def validate_fred_api_key(cls) -> bool:
        """Validate if FRED API key is properly configured"""
        if not cls.FRED_API_KEY:
            return False
        if cls.FRED_API_KEY == 'your-fred-api-key-here':
            return False
        return True
    
    @classmethod
    def get_fred_api_key(cls) -> Optional[str]:
        """Get FRED API key with validation"""
        if cls.validate_fred_api_key():
            return cls.FRED_API_KEY
        return None

def setup_fred_api_key():
    """Helper function to guide users in setting up FRED API key"""
    print("=" * 60)
    print("FRED ML - API Key Setup")
    print("=" * 60)
    print()
    print("To use real FRED data, you need to:")
    print("1. Get a free API key from: https://fred.stlouisfed.org/docs/api/api_key.html")
    print("2. Set the environment variable:")
    print("   export FRED_API_KEY='your-api-key-here'")
    print()
    print("Or create a .env file in the project root with:")
    print("FRED_API_KEY=your-api-key-here")
    print()
    print("The application will work with demo data if no API key is provided.")
    print("=" * 60)

if __name__ == "__main__":
    setup_fred_api_key() 