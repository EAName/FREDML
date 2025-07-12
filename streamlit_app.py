#!/usr/bin/env python3
"""Diagnostic entry‐point for Streamlit Cloud deployment."""
import os, sys
import streamlit as st
from dotenv import load_dotenv

# 1. Load .env locally (no‐op in Cloud)…
load_dotenv()

# 2. Grab the key from env OR Cloud secrets
fred_key = os.getenv("FRED_API_KEY") or st.secrets.get("FRED_API_KEY")
if not fred_key:
    st.error("❌ FRED API not available. Please configure your FRED_API_KEY.")
    st.stop()

# 3. Propagate it into the actual env & secret namespace  
os.environ["FRED_API_KEY"] = fred_key
st.secrets["FRED_API_KEY"] = fred_key  # so any direct st.secrets lookup also works

# 4. Now hook up your frontend code
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "frontend"))
from app import main as app_main

# Call the main function directly for Streamlit Cloud
app_main() 