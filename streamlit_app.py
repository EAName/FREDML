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

# Add the current directory to the path as well
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

main = None  # Initialize main as None

try:
    # Import only the main function to avoid loading unnecessary modules
    from frontend.app import main
    print("Successfully imported main function from frontend.app")
except ImportError as e:
    print(f"Error importing from frontend.app: {e}")
    try:
        # Fallback: try importing directly
        from app import main
        print("Successfully imported main function from app")
    except ImportError as e2:
        print(f"Error importing from app: {e2}")
        # Last resort: define a simple main function
        import streamlit as st
        def main():
            st.error("Failed to import main application. Please check the deployment.")
            st.info("Contact support if this issue persists.")
        print("Using fallback main function")

# Run the main function directly
if __name__ == "__main__":
    if main is not None:
        main()
    else:
        import streamlit as st
        st.error("Failed to import main application. Please check the deployment.")
        st.info("Contact support if this issue persists.") 