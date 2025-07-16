#!/usr/bin/env python3
"""Streamlit-native entry point for Streamlit Cloud deployment."""
import streamlit as st
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('.'))

# **no** load_dotenv() here
fred_key = st.secrets.get("FRED_API_KEY", "")
if not fred_key:
    st.error("❌ FRED API key not found in Streamlit secrets.")
    st.stop()

# make it available to downstream code
os.environ["FRED_API_KEY"] = fred_key

# now import and run your real app
try:
    from frontend.app import main as app_main
    app_main()
except ImportError as e:
    st.error(f"❌ Failed to import main function: {e}")
    st.info("Please check that the frontend/app.py file exists and contains a main() function.")
    st.stop() 