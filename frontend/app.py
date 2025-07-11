#!/usr/bin/env python3
"""
FRED ML - Streamlit Frontend
Interactive web application for economic data analysis
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import boto3
import json
from datetime import datetime, timedelta
import requests
import os
from typing import Dict, List, Optional

# Page configuration
st.set_page_config(
    page_title="FRED ML - Economic Data Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize AWS clients
@st.cache_resource
def init_aws_clients():
    """Initialize AWS clients for S3 and Lambda"""
    try:
        s3_client = boto3.client('s3')
        lambda_client = boto3.client('lambda')
        return s3_client, lambda_client
    except Exception as e:
        st.error(f"Failed to initialize AWS clients: {e}")
        return None, None

# Load configuration
@st.cache_data
def load_config():
    """Load application configuration"""
    return {
        's3_bucket': os.getenv('S3_BUCKET', 'fredmlv1'),
        'lambda_function': os.getenv('LAMBDA_FUNCTION', 'fred-ml-processor'),
        'api_endpoint': os.getenv('API_ENDPOINT', 'http://localhost:8000')
    }

def get_available_reports(s3_client, bucket_name: str) -> List[Dict]:
    """Get list of available reports from S3"""
    try:
        response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix='reports/'
        )
        
        reports = []
        if 'Contents' in response:
            for obj in response['Contents']:
                if obj['Key'].endswith('.json'):
                    reports.append({
                        'key': obj['Key'],
                        'last_modified': obj['LastModified'],
                        'size': obj['Size']
                    })
        
        return sorted(reports, key=lambda x: x['last_modified'], reverse=True)
    except Exception as e:
        st.error(f"Failed to load reports: {e}")
        return []

def get_report_data(s3_client, bucket_name: str, report_key: str) -> Optional[Dict]:
    """Get report data from S3"""
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=report_key)
        data = json.loads(response['Body'].read().decode('utf-8'))
        return data
    except Exception as e:
        st.error(f"Failed to load report data: {e}")
        return None

def trigger_lambda_analysis(lambda_client, function_name: str, payload: Dict) -> bool:
    """Trigger Lambda function for analysis"""
    try:
        response = lambda_client.invoke(
            FunctionName=function_name,
            InvocationType='Event',  # Asynchronous
            Payload=json.dumps(payload)
        )
        return response['StatusCode'] == 202
    except Exception as e:
        st.error(f"Failed to trigger analysis: {e}")
        return False

def create_time_series_plot(df: pd.DataFrame, title: str = "Economic Indicators"):
    """Create interactive time series plot"""
    fig = go.Figure()
    
    for column in df.columns:
        if column != 'Date':
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[column],
                    mode='lines',
                    name=column,
                    line=dict(width=2)
                )
            )
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Value",
        hovermode='x unified',
        height=500
    )
    
    return fig

def create_correlation_heatmap(df: pd.DataFrame):
    """Create correlation heatmap"""
    corr_matrix = df.corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        title="Correlation Matrix"
    )
    
    return fig

def main():
    """Main Streamlit application"""
    
    # Initialize AWS clients
    s3_client, lambda_client = init_aws_clients()
    config = load_config()
    
    # Sidebar
    st.sidebar.title("FRED ML Dashboard")
    st.sidebar.markdown("---")
    
    # Navigation
    page = st.sidebar.selectbox(
        "Navigation",
        ["📊 Dashboard", "📈 Analysis", "📋 Reports", "⚙️ Settings"]
    )
    
    if page == "📊 Dashboard":
        show_dashboard(s3_client, config)
    elif page == "📈 Analysis":
        show_analysis_page(lambda_client, config)
    elif page == "📋 Reports":
        show_reports_page(s3_client, config)
    elif page == "⚙️ Settings":
        show_settings_page(config)

def show_dashboard(s3_client, config):
    """Show main dashboard"""
    st.title("📊 FRED ML Dashboard")
    st.markdown("Economic Data Analysis Platform")
    
    # Get latest report
    reports = get_available_reports(s3_client, config['s3_bucket'])
    
    if reports:
        latest_report = reports[0]
        report_data = get_report_data(s3_client, config['s3_bucket'], latest_report['key'])
        
        if report_data:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Latest Analysis",
                    latest_report['last_modified'].strftime("%Y-%m-%d"),
                    f"Updated {latest_report['last_modified'].strftime('%H:%M')}"
                )
            
            with col2:
                st.metric(
                    "Data Points",
                    report_data.get('total_observations', 'N/A'),
                    "Economic indicators"
                )
            
            with col3:
                st.metric(
                    "Time Range",
                    f"{report_data.get('start_date', 'N/A')} - {report_data.get('end_date', 'N/A')}",
                    "Analysis period"
                )
            
            # Show latest data visualization
            if 'data' in report_data and report_data['data']:
                df = pd.DataFrame(report_data['data'])
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
                
                st.subheader("Latest Economic Indicators")
                fig = create_time_series_plot(df)
                st.plotly_chart(fig, use_container_width=True)
                
                # Correlation matrix
                st.subheader("Correlation Analysis")
                corr_fig = create_correlation_heatmap(df)
                st.plotly_chart(corr_fig, use_container_width=True)
        else:
            st.warning("No report data available")
    else:
        st.info("No reports available. Run an analysis to generate reports.")

def show_analysis_page(lambda_client, config):
    """Show analysis configuration page"""
    st.title("📈 Economic Data Analysis")
    
    # Analysis parameters
    st.subheader("Analysis Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Economic indicators selection
        indicators = [
            "GDP", "UNRATE", "CPIAUCSL", "FEDFUNDS", "DGS10",
            "DEXUSEU", "PAYEMS", "INDPRO", "M2SL", "PCE"
        ]
        
        selected_indicators = st.multiselect(
            "Select Economic Indicators",
            indicators,
            default=["GDP", "UNRATE", "CPIAUCSL"]
        )
    
    with col2:
        # Date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*2)  # 2 years
        
        start_date_input = st.date_input(
            "Start Date",
            value=start_date,
            max_value=end_date
        )
        
        end_date_input = st.date_input(
            "End Date",
            value=end_date,
            max_value=end_date
        )
    
    # Analysis options
    st.subheader("Analysis Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        include_visualizations = st.checkbox("Generate Visualizations", value=True)
        include_correlation = st.checkbox("Correlation Analysis", value=True)
    
    with col2:
        include_forecasting = st.checkbox("Time Series Forecasting", value=False)
        include_statistics = st.checkbox("Statistical Summary", value=True)
    
    # Run analysis button
    if st.button("🚀 Run Analysis", type="primary"):
        if not selected_indicators:
            st.error("Please select at least one economic indicator")
        elif start_date_input >= end_date_input:
            st.error("Start date must be before end date")
        else:
            with st.spinner("Running analysis..."):
                payload = {
                    'indicators': selected_indicators,
                    'start_date': start_date_input.strftime('%Y-%m-%d'),
                    'end_date': end_date_input.strftime('%Y-%m-%d'),
                    'options': {
                        'visualizations': include_visualizations,
                        'correlation': include_correlation,
                        'forecasting': include_forecasting,
                        'statistics': include_statistics
                    }
                }
                
                success = trigger_lambda_analysis(lambda_client, config['lambda_function'], payload)
                
                if success:
                    st.success("Analysis triggered successfully! Check the Reports page for results.")
                else:
                    st.error("Failed to trigger analysis")

def show_reports_page(s3_client, config):
    """Show reports page"""
    st.title("📋 Analysis Reports")
    
    reports = get_available_reports(s3_client, config['s3_bucket'])
    
    if reports:
        st.subheader(f"Available Reports ({len(reports)})")
        
        for i, report in enumerate(reports):
            with st.expander(f"Report {i+1} - {report['last_modified'].strftime('%Y-%m-%d %H:%M')}"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(f"**File:** {report['key']}")
                    st.write(f"**Size:** {report['size']} bytes")
                    st.write(f"**Last Modified:** {report['last_modified']}")
                
                with col2:
                    if st.button(f"View Report {i+1}", key=f"view_{i}"):
                        report_data = get_report_data(s3_client, config['s3_bucket'], report['key'])
                        if report_data:
                            st.json(report_data)
    else:
        st.info("No reports available. Run an analysis to generate reports.")

def show_settings_page(config):
    """Show settings page"""
    st.title("⚙️ Settings")
    
    st.subheader("Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**S3 Bucket:** {config['s3_bucket']}")
        st.write(f"**Lambda Function:** {config['lambda_function']}")
    
    with col2:
        st.write(f"**API Endpoint:** {config['api_endpoint']}")
    
    st.subheader("Environment Variables")
    st.code(f"""
S3_BUCKET={config['s3_bucket']}
LAMBDA_FUNCTION={config['lambda_function']}
API_ENDPOINT={config['api_endpoint']}
    """)

if __name__ == "__main__":
    main() 