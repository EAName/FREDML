#!/usr/bin/env python3
"""Streamlit-native entry point for Streamlit Cloud deployment."""
import os, sys
import streamlit as st

# Streamlit-native solution: Use st.secrets directly
fred_key = os.getenv("FRED_API_KEY") or st.secrets["FRED_API_KEY"]
if not fred_key:
    st.error("❌ FRED API key not found. Configure it in Streamlit Cloud Secrets.")
    st.info("Available environment variables: " + str(list(os.environ.keys())))
    st.info("Available secrets keys: " + str(list(st.secrets.keys())))
    st.stop()

# Set the environment variable for the frontend app
os.environ["FRED_API_KEY"] = fred_key
print(f"DEBUG: Set FRED_API_KEY in environment = {os.environ.get('FRED_API_KEY')}")

# Now hook up your frontend code
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "frontend"))
from app import main as app_main

# Call the main function directly for Streamlit Cloud
app_main() 