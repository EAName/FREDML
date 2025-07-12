#!/usr/bin/env python3
"""Streamlit-native entry point for Streamlit Cloud deployment."""
import streamlit as st, os

# **no** load_dotenv() here
fred_key = st.secrets["FRED_API_KEY"]
if not fred_key:
    st.error("❌ FRED API key not found in Streamlit secrets.")
    st.stop()

# make it available to downstream code
os.environ["FRED_API_KEY"] = fred_key

# now import and run your real app
from frontend.app import main as app_main
app_main() 