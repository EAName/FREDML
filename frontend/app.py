#!/usr/bin/env python3
"""
FRED ML - Enterprise Economic Analytics Platform
Professional think tank interface for comprehensive economic data analysis
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
import sys
from typing import Dict, List, Optional
from pathlib import Path

# Add src to path for analytics modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import analytics modules
try:
    from src.analysis.comprehensive_analytics import ComprehensiveAnalytics
    from src.core.enhanced_fred_client import EnhancedFREDClient
    from config.settings import FRED_API_KEY
    ANALYTICS_AVAILABLE = True
except ImportError:
    ANALYTICS_AVAILABLE = False
    st.warning("Advanced analytics modules not available. Running in basic mode.")

# Page configuration
st.set_page_config(
    page_title="FRED ML - Economic Analytics Platform",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enterprise styling
st.markdown("""
<style>
    /* Main styling */
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #1e3c72;
        margin-bottom: 1rem;
    }
    
    .analysis-section {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #e9ecef;
    }
    
    .sidebar .sidebar-content {
        background: #2c3e50;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    
    .stButton > button:hover {
        background: linear-gradient(90deg, #2a5298 0%, #1e3c72 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    .success-message {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    
    .warning-message {
        background: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #ffeaa7;
        margin: 1rem 0;
    }
    
    .info-message {
        background: #d1ecf1;
        color: #0c5460;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #bee5eb;
        margin: 1rem 0;
    }
    
    .chart-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    
    .tabs-container {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

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
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, column in enumerate(df.columns):
        if column != 'Date':
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[column],
                    mode='lines',
                    name=column,
                    line=dict(width=2, color=colors[i % len(colors)]),
                    hovertemplate='<b>%{x}</b><br>%{y:.2f}<extra></extra>'
                )
            )
    
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=20)),
        xaxis_title="Date",
        yaxis_title="Value",
        hovermode='x unified',
        height=500,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12)
    )
    
    return fig

def create_correlation_heatmap(df: pd.DataFrame):
    """Create correlation heatmap"""
    corr_matrix = df.corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        title="Correlation Matrix",
        color_continuous_scale='RdBu_r',
        center=0
    )
    
    fig.update_layout(
        title=dict(x=0.5, font=dict(size=20)),
        height=500,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig

def create_forecast_plot(historical_data, forecast_data, title="Forecast"):
    """Create forecast plot with confidence intervals"""
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=historical_data.index,
        y=historical_data.values,
        mode='lines',
        name='Historical',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # Forecast
    if 'forecast' in forecast_data:
        forecast_values = forecast_data['forecast']
        forecast_index = pd.date_range(
            start=historical_data.index[-1] + pd.DateOffset(months=3),
            periods=len(forecast_values),
            freq='Q'
        )
        
        fig.add_trace(go.Scatter(
            x=forecast_index,
            y=forecast_values,
            mode='lines',
            name='Forecast',
            line=dict(color='#ff7f0e', width=2, dash='dash')
        ))
        
        # Confidence intervals
        if 'confidence_intervals' in forecast_data:
            ci = forecast_data['confidence_intervals']
            if 'lower' in ci.columns and 'upper' in ci.columns:
                fig.add_trace(go.Scatter(
                    x=forecast_index,
                    y=ci['upper'],
                    mode='lines',
                    name='Upper CI',
                    line=dict(color='rgba(255,127,14,0.3)', width=1),
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=forecast_index,
                    y=ci['lower'],
                    mode='lines',
                    fill='tonexty',
                    name='Confidence Interval',
                    line=dict(color='rgba(255,127,14,0.3)', width=1)
                ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=20)),
        xaxis_title="Date",
        yaxis_title="Value",
        height=500,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig

def main():
    """Main Streamlit application"""
    
    # Initialize AWS clients
    s3_client, lambda_client = init_aws_clients()
    config = load_config()
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <h2>🏛️ FRED ML</h2>
            <p style="color: #666; font-size: 0.9rem;">Economic Analytics Platform</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Navigation
        page = st.selectbox(
            "Navigation",
            ["📊 Executive Dashboard", "🔮 Advanced Analytics", "📈 Economic Indicators", "📋 Reports & Insights", "⚙️ Configuration"]
        )
    
    if page == "📊 Executive Dashboard":
        show_executive_dashboard(s3_client, config)
    elif page == "🔮 Advanced Analytics":
        show_advanced_analytics_page(config)
    elif page == "📈 Economic Indicators":
        show_indicators_page(s3_client, config)
    elif page == "📋 Reports & Insights":
        show_reports_page(s3_client, config)
    elif page == "⚙️ Configuration":
        show_configuration_page(config)

def show_executive_dashboard(s3_client, config):
    """Show executive dashboard with key metrics"""
    st.markdown("""
    <div class="main-header">
        <h1>📊 Executive Dashboard</h1>
        <p>Comprehensive Economic Analytics & Insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>📈 GDP Growth</h3>
            <h2>2.1%</h2>
            <p>Q4 2024</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🏭 Industrial Production</h3>
            <h2>+0.8%</h2>
            <p>Monthly Change</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>💰 Inflation Rate</h3>
            <h2>3.2%</h2>
            <p>Annual Rate</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>💼 Unemployment</h3>
            <h2>3.7%</h2>
            <p>Current Rate</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Recent analysis section
    st.markdown("""
    <div class="analysis-section">
        <h3>📊 Recent Analysis</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Get latest report
    reports = get_available_reports(s3_client, config['s3_bucket'])
    
    if reports:
        latest_report = reports[0]
        report_data = get_report_data(s3_client, config['s3_bucket'], latest_report['key'])
        
        if report_data:
            # Show latest data visualization
            if 'data' in report_data and report_data['data']:
                df = pd.DataFrame(report_data['data'])
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    <div class="chart-container">
                        <h4>Economic Indicators Trend</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    fig = create_time_series_plot(df)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("""
                    <div class="chart-container">
                        <h4>Correlation Analysis</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    corr_fig = create_correlation_heatmap(df)
                    st.plotly_chart(corr_fig, use_container_width=True)
        else:
            st.warning("No report data available")
    else:
        st.info("No reports available. Run an analysis to generate reports.")

def show_advanced_analytics_page(config):
    """Show advanced analytics page with comprehensive analysis capabilities"""
    st.markdown("""
    <div class="main-header">
        <h1>🔮 Advanced Analytics</h1>
        <p>Comprehensive Economic Modeling & Forecasting</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not ANALYTICS_AVAILABLE:
        st.error("Advanced analytics modules not available. Please install required dependencies.")
        return
    
    # Analysis configuration
    st.markdown("""
    <div class="analysis-section">
        <h3>📋 Analysis Configuration</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Economic indicators selection
        indicators = [
            "GDPC1", "INDPRO", "RSAFS", "CPIAUCSL", "FEDFUNDS", "DGS10",
            "TCU", "PAYEMS", "PCE", "M2SL", "DEXUSEU", "UNRATE"
        ]
        
        selected_indicators = st.multiselect(
            "Select Economic Indicators",
            indicators,
            default=["GDPC1", "INDPRO", "RSAFS"]
        )
        
        # Date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*5)  # 5 years
        
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
    
    with col2:
        # Analysis options
        forecast_periods = st.slider(
            "Forecast Periods",
            min_value=1,
            max_value=12,
            value=4,
            help="Number of periods to forecast"
        )
        
        include_visualizations = st.checkbox(
            "Generate Visualizations",
            value=True,
            help="Create charts and graphs"
        )
        
        analysis_type = st.selectbox(
            "Analysis Type",
            ["Comprehensive", "Forecasting Only", "Segmentation Only", "Statistical Only"],
            help="Type of analysis to perform"
        )
    
    # Run analysis button
    if st.button("🚀 Run Advanced Analysis", type="primary"):
        if not selected_indicators:
            st.error("Please select at least one economic indicator.")
            return
        
        if not FRED_API_KEY:
            st.error("FRED API key not configured. Please set FRED_API_KEY environment variable.")
            return
        
        # Show progress
        with st.spinner("Running comprehensive analysis..."):
            try:
                # Initialize analytics
                analytics = ComprehensiveAnalytics(FRED_API_KEY, output_dir="data/exports/streamlit")
                
                # Run analysis
                results = analytics.run_complete_analysis(
                    indicators=selected_indicators,
                    start_date=start_date_input.strftime('%Y-%m-%d'),
                    end_date=end_date_input.strftime('%Y-%m-%d'),
                    forecast_periods=forecast_periods,
                    include_visualizations=include_visualizations
                )
                
                st.success("✅ Analysis completed successfully!")
                
                # Display results
                display_analysis_results(results)
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {e}")

def display_analysis_results(results):
    """Display comprehensive analysis results"""
    st.markdown("""
    <div class="analysis-section">
        <h3>📊 Analysis Results</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different result types
    tab1, tab2, tab3, tab4 = st.tabs(["🔮 Forecasting", "🎯 Segmentation", "📈 Statistical", "💡 Insights"])
    
    with tab1:
        if 'forecasting' in results:
            st.subheader("Forecasting Results")
            forecasting_results = results['forecasting']
            
            for indicator, result in forecasting_results.items():
                if 'error' not in result:
                    backtest = result.get('backtest', {})
                    if 'error' not in backtest:
                        mape = backtest.get('mape', 0)
                        rmse = backtest.get('rmse', 0)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(f"{indicator} MAPE", f"{mape:.2f}%")
                        with col2:
                            st.metric(f"{indicator} RMSE", f"{rmse:.4f}")
    
    with tab2:
        if 'segmentation' in results:
            st.subheader("Segmentation Results")
            segmentation_results = results['segmentation']
            
            if 'time_period_clusters' in segmentation_results:
                time_clusters = segmentation_results['time_period_clusters']
                if 'error' not in time_clusters:
                    n_clusters = time_clusters.get('n_clusters', 0)
                    st.info(f"Time periods clustered into {n_clusters} economic regimes")
            
            if 'series_clusters' in segmentation_results:
                series_clusters = segmentation_results['series_clusters']
                if 'error' not in series_clusters:
                    n_clusters = series_clusters.get('n_clusters', 0)
                    st.info(f"Economic series clustered into {n_clusters} groups")
    
    with tab3:
        if 'statistical_modeling' in results:
            st.subheader("Statistical Analysis Results")
            stat_results = results['statistical_modeling']
            
            if 'correlation' in stat_results:
                corr_results = stat_results['correlation']
                significant_correlations = corr_results.get('significant_correlations', [])
                st.info(f"Found {len(significant_correlations)} significant correlations")
    
    with tab4:
        if 'insights' in results:
            st.subheader("Key Insights")
            insights = results['insights']
            
            for finding in insights.get('key_findings', []):
                st.write(f"• {finding}")

def show_indicators_page(s3_client, config):
    """Show economic indicators page"""
    st.markdown("""
    <div class="main-header">
        <h1>📈 Economic Indicators</h1>
        <p>Real-time Economic Data & Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Indicators overview
    indicators_info = {
        "GDPC1": {"name": "Real GDP", "description": "Real Gross Domestic Product", "frequency": "Quarterly"},
        "INDPRO": {"name": "Industrial Production", "description": "Industrial Production Index", "frequency": "Monthly"},
        "RSAFS": {"name": "Retail Sales", "description": "Retail Sales", "frequency": "Monthly"},
        "CPIAUCSL": {"name": "Consumer Price Index", "description": "Inflation measure", "frequency": "Monthly"},
        "FEDFUNDS": {"name": "Federal Funds Rate", "description": "Target interest rate", "frequency": "Daily"},
        "DGS10": {"name": "10-Year Treasury", "description": "Government bond yield", "frequency": "Daily"}
    }
    
    # Display indicators in cards
    cols = st.columns(3)
    for i, (code, info) in enumerate(indicators_info.items()):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{info['name']}</h3>
                <p><strong>Code:</strong> {code}</p>
                <p><strong>Frequency:</strong> {info['frequency']}</p>
                <p>{info['description']}</p>
            </div>
            """, unsafe_allow_html=True)

def show_reports_page(s3_client, config):
    """Show reports and insights page"""
    st.markdown("""
    <div class="main-header">
        <h1>📋 Reports & Insights</h1>
        <p>Comprehensive Analysis Reports</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get available reports
    reports = get_available_reports(s3_client, config['s3_bucket'])
    
    if reports:
        st.subheader("Available Reports")
        
        for report in reports[:5]:  # Show last 5 reports
            with st.expander(f"Report: {report['key']} - {report['last_modified'].strftime('%Y-%m-%d %H:%M')}"):
                report_data = get_report_data(s3_client, config['s3_bucket'], report['key'])
                if report_data:
                    st.json(report_data)
    else:
        st.info("No reports available. Run an analysis to generate reports.")

def show_configuration_page(config):
    """Show configuration page"""
    st.markdown("""
    <div class="main-header">
        <h1>⚙️ Configuration</h1>
        <p>System Settings & Configuration</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("System Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**AWS Configuration**")
        st.write(f"S3 Bucket: {config['s3_bucket']}")
        st.write(f"Lambda Function: {config['lambda_function']}")
    
    with col2:
        st.write("**API Configuration**")
        st.write(f"API Endpoint: {config['api_endpoint']}")
        st.write(f"Analytics Available: {ANALYTICS_AVAILABLE}")

if __name__ == "__main__":
    main() 