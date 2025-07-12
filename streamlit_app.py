#!/usr/bin/env python3
"""Diagnostic entry‐point for Streamlit Cloud deployment."""
import os, sys

# CRITICAL: Set environment variable BEFORE importing streamlit
# This ensures the key is available when the frontend app imports

# 1. Try to get the key from environment first
fred_key = os.getenv("FRED_API_KEY")

# 2. If not in environment, we'll get it from Streamlit secrets later
if not fred_key:
    # Import streamlit only when needed
    import streamlit as st
    fred_key = st.secrets.get("FRED_API_KEY")

print(f"DEBUG: FRED_API_KEY from os.getenv = {os.getenv('FRED_API_KEY')}")
print(f"DEBUG: Final fred_key = {fred_key}")

# 3. Set the environment variable IMMEDIATELY
if fred_key:
    os.environ["FRED_API_KEY"] = fred_key
    print(f"DEBUG: Set FRED_API_KEY in environment = {os.environ.get('FRED_API_KEY')}")
else:
    print("DEBUG: No FRED API key found!")

# 4. Now import streamlit and frontend code
import streamlit as st
from dotenv import load_dotenv

# Load .env locally (no‐op in Cloud)…
load_dotenv()

# 5. Double-check the key is available
if not os.getenv("FRED_API_KEY"):
    st.error("❌ FRED API not available. Please configure your FRED_API_KEY.")
    st.info("Available environment variables: " + str(list(os.environ.keys())))
    st.info("Available secrets keys: " + str(list(st.secrets.keys())))
    st.stop()

# 6. Now hook up your frontend code
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "frontend"))
from app import main as app_main

# Call the main function directly for Streamlit Cloud
app_main() 