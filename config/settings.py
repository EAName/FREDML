import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# FRED API Configuration
FRED_API_KEY = os.getenv("FRED_API_KEY")

# Data settings
DEFAULT_START_DATE = "2010-01-01"
DEFAULT_END_DATE = "2024-01-01"

# Output settings
OUTPUT_DIR = "data"
PLOTS_DIR = "plots" 