#!/usr/bin/env python3
"""Entry‐point for Streamlit Cloud deployment."""
import os, sys
from dotenv import load_dotenv

# 1) Load .env (locally) and merge in Streamlit secrets
load_dotenv()
from streamlit import secrets  # for Cloud

# 2) Ensure our code is importable
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "frontend"))
sys.path.insert(0, HERE)

# 3) Get the FRED key from env or Streamlit secrets
fred_key = os.getenv("FRED_API_KEY") or secrets.get("FRED_API_KEY")
if not fred_key:
    import streamlit as st
    st.error("❌ FRED API not available. Please configure your FRED_API_KEY.")
    st.stop()

# 4) Import and run your real app
from frontend.app import main as app_main

if __name__ == "__main__":
    app_main() 