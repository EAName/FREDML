#!/usr/bin/env python3
"""
FRED ML - Economic Analytics Platform
Streamlit Cloud Deployment Entry Point
"""

import sys
import os

# Add the frontend directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
frontend_dir = os.path.join(current_dir, 'frontend')
if frontend_dir not in sys.path:
    sys.path.insert(0, frontend_dir)

# Import only the main function to avoid loading unnecessary modules
from app import main

# Run the main function directly
if __name__ == "__main__":
    main() 