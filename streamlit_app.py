#!/usr/bin/env python3
"""Diagnostic wrapper to verify Streamlit secrets."""
import os, streamlit as st

st.write("🔑 os.environ:", {k: v for k, v in os.environ.items() if "FRED" in k})
st.write("🔑 st.secrets:", list(st.secrets.keys()))
st.write("🔑 st.secrets FRED_API_KEY:", st.secrets.get("FRED_API_KEY", "NOT_FOUND"))
st.stop() 