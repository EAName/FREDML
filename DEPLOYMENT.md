# FRED ML - Streamlit Cloud Deployment Guide

## Overview
This guide explains how to deploy the FRED ML Economic Analytics Platform to Streamlit Cloud for free.

## Prerequisites
1. GitHub account
2. Streamlit Cloud account (free at https://share.streamlit.io/)

## Deployment Steps

### 1. Push to GitHub
```bash
git add .
git commit -m "Prepare for Streamlit Cloud deployment"
git push origin main
```

### 2. Deploy to Streamlit Cloud
1. Go to https://share.streamlit.io/
2. Sign in with GitHub
3. Click "New app"
4. Select your repository: `your-username/FRED_ML`
5. Set the main file path: `streamlit_app.py`
6. Click "Deploy"

### 3. Configure Environment Variables
In Streamlit Cloud dashboard:
1. Go to your app settings
2. Add these environment variables:
   - `FRED_API_KEY`: Your FRED API key
   - `AWS_ACCESS_KEY_ID`: Your AWS access key
   - `AWS_SECRET_ACCESS_KEY`: Your AWS secret key
   - `AWS_REGION`: us-east-1

### 4. Access Your App
Your app will be available at: `https://your-app-name-your-username.streamlit.app`

## Features Available in Deployment
- ✅ Real FRED API data integration
- ✅ Advanced analytics and forecasting
- ✅ Professional enterprise-grade UI
- ✅ AWS S3 integration (if credentials provided)
- ✅ Local storage fallback
- ✅ Comprehensive download capabilities

## Troubleshooting
- If you see import errors, check that all dependencies are in `requirements.txt`
- If AWS features don't work, verify your AWS credentials in environment variables
- If FRED API doesn't work, check your FRED API key

## Security Notes
- Never commit `.env` files to GitHub
- Use Streamlit Cloud's environment variables for sensitive data
- AWS credentials are automatically secured by Streamlit Cloud 