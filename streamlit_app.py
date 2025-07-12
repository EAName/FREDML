#!/usr/bin/env python3
"""Diagnostic entry‐point for Streamlit Cloud deployment."""
import os, sys
import streamlit as st
from dotenv import load_dotenv

# 1) Load .env (locally) and merge in Streamlit secrets
load_dotenv()  # Local .env, no-op on Cloud

st.write("🌱 os.environ keys with 'FRED': ", [k for k in os.environ if "FRED" in k])
st.write("🌱 st.secrets keys: ", list(st.secrets.keys()))

# Now fetch the key from both places
env_key = os.getenv("FRED_API_KEY")
secrets_key = st.secrets.get("FRED_API_KEY")
st.write("🌱 os.getenv FRED_API_KEY:", env_key or "‹None›")
st.write("🌱 st.secrets FRED_API_KEY:", secrets_key or "‹None›")

# Test FRED API call if we have a key
fred_key = env_key or secrets_key
if fred_key:
    try:
        from fredapi import Fred
        fred = Fred(api_key=fred_key)
        
        # Quick sanity check:
        pts = fred.get_series("GDP", observation_start="2020-01-01", observation_end="2020-01-01")
        st.write("✅ Fetched test point:", pts.iloc[0])
        st.success("🎉 FRED API connection successful!")
        
        # If we get here, the API works - let's try the real app
        st.write("🚀 Attempting to load real app...")
        
        # 2) Ensure our code is importable
        HERE = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, os.path.join(HERE, "frontend"))
        sys.path.insert(0, HERE)
        
        # 3) Import and run your real app
        from frontend.app import main as app_main
        app_main()
        
    except Exception as e:
        st.error(f"🚨 fredapi call failed: {e}")
        st.stop()
else:
    st.error("❌ No FRED API key found in environment or secrets")
    st.info("Please configure FRED_API_KEY in Streamlit Cloud secrets")
    st.stop() 