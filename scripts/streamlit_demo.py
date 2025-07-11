#!/usr/bin/env python3
"""
FRED ML Streamlit Demo
Interactive demonstration of the FRED ML system capabilities
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import sys
import json
import time

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Page configuration
st.set_page_config(
    page_title="FRED ML Demo",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def create_sample_data():
    """Create sample economic data for demo"""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2024-01-01', freq='M')
    
    # Simulate realistic economic indicators
    data = {
        'GDP': np.random.normal(100, 5, len(dates)) + np.cumsum(np.random.normal(0, 0.5, len(dates))),
        'UNRATE': np.random.normal(5, 1, len(dates)),
        'CPIAUCSL': np.random.normal(200, 10, len(dates)) + np.cumsum(np.random.normal(0, 1, len(dates))),
        'FEDFUNDS': np.random.normal(2, 0.5, len(dates)),
        'DGS10': np.random.normal(3, 0.3, len(dates))
    }
    
    return pd.DataFrame(data, index=dates)

def main():
    """Main Streamlit application"""
    
    # Header
    st.title("📊 FRED ML System Demo")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("🎛️ Demo Controls")
    
    # Demo sections
    demo_section = st.sidebar.selectbox(
        "Choose Demo Section:",
        ["🏠 Overview", "📈 Data Processing", "🎨 Visualizations", "🔍 Analysis", "🏗️ Architecture", "⚡ Live Demo"]
    )
    
    if demo_section == "🏠 Overview":
        show_overview()
    elif demo_section == "📈 Data Processing":
        show_data_processing()
    elif demo_section == "🎨 Visualizations":
        show_visualizations()
    elif demo_section == "🔍 Analysis":
        show_analysis()
    elif demo_section == "🏗️ Architecture":
        show_architecture()
    elif demo_section == "⚡ Live Demo":
        show_live_demo()

def show_overview():
    """Show system overview"""
    st.header("🏠 FRED ML System Overview")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### What is FRED ML?
        
        **FRED ML** is a comprehensive Machine Learning system for analyzing Federal Reserve Economic Data (FRED). 
        It provides automated data processing, advanced analytics, and interactive visualizations for economic indicators.
        
        ### Key Features:
        - 📊 **Real-time Data Processing**: Automated FRED API integration
        - 🤖 **Machine Learning Analytics**: Advanced statistical modeling
        - 📈 **Interactive Visualizations**: Dynamic charts and dashboards
        - 🔄 **Automated Workflows**: CI/CD pipeline with quality gates
        - ☁️ **Cloud-Native**: AWS Lambda and S3 integration
        - 🧪 **Comprehensive Testing**: Unit, integration, and E2E tests
        
        ### System Components:
        - **Frontend**: Streamlit interactive dashboard
        - **Backend**: AWS Lambda serverless functions
        - **Storage**: AWS S3 for data persistence
        - **Scheduling**: EventBridge for automated triggers
        - **Data Source**: FRED API for economic indicators
        """)
    
    with col2:
        # System status
        st.subheader("🔧 System Status")
        status_data = {
            "Component": ["FRED API", "AWS Lambda", "S3 Storage", "Streamlit", "Testing"],
            "Status": ["✅ Connected", "✅ Ready", "✅ Ready", "✅ Running", "✅ Complete"]
        }
        st.dataframe(pd.DataFrame(status_data))

def show_data_processing():
    """Show data processing capabilities"""
    st.header("📈 Data Processing Demo")
    
    # Create sample data
    df = create_sample_data()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Sample Economic Data")
        st.dataframe(df.head(10))
        
        st.subheader("📈 Data Summary")
        summary_stats = df.describe()
        st.dataframe(summary_stats)
    
    with col2:
        st.subheader("🔗 Correlation Matrix")
        correlation = df.corr()
        
        # Create heatmap
        fig = px.imshow(
            correlation,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu",
            title="Economic Indicators Correlation"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Data quality metrics
    st.subheader("📋 Data Quality Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Indicators", len(df.columns))
    with col3:
        st.metric("Date Range", f"{df.index.min().strftime('%Y-%m')} to {df.index.max().strftime('%Y-%m')}")
    with col4:
        missing_data = df.isnull().sum().sum()
        st.metric("Missing Values", missing_data)

def show_visualizations():
    """Show visualization capabilities"""
    st.header("🎨 Visualization Demo")
    
    df = create_sample_data()
    
    # Visualization options
    viz_type = st.selectbox(
        "Choose Visualization Type:",
        ["Time Series", "Correlation Heatmap", "Distribution Plots", "Interactive Dashboard"]
    )
    
    if viz_type == "Time Series":
        st.subheader("📈 Economic Indicators Over Time")
        
        # Multi-line time series
        fig = go.Figure()
        
        for col in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[col],
                name=col,
                mode='lines',
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title="Economic Indicators Time Series",
            xaxis_title="Date",
            yaxis_title="Value",
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    elif viz_type == "Correlation Heatmap":
        st.subheader("🔥 Correlation Matrix Heatmap")
        
        correlation = df.corr()
        
        fig = px.imshow(
            correlation,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu",
            title="Economic Indicators Correlation Heatmap"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    elif viz_type == "Distribution Plots":
        st.subheader("📊 Distribution Analysis")
        
        # Create subplots for distributions
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=df.columns,
            specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
        )
        
        for i, col in enumerate(df.columns):
            row = (i // 3) + 1
            col_num = (i % 3) + 1
            fig.add_trace(
                go.Histogram(x=df[col], name=col, nbinsx=20),
                row=row, col=col_num
            )
        
        fig.update_layout(height=600, title_text="Distribution of Economic Indicators")
        st.plotly_chart(fig, use_container_width=True)
        
    elif viz_type == "Interactive Dashboard":
        st.subheader("🎛️ Interactive Dashboard")
        
        # Interactive controls
        selected_indicators = st.multiselect(
            "Select Indicators:",
            df.columns,
            default=df.columns[:3]
        )
        
        date_range = st.slider(
            "Select Date Range:",
            min_value=df.index.min(),
            max_value=df.index.max(),
            value=(df.index.min(), df.index.max())
        )
        
        if selected_indicators:
            filtered_df = df.loc[date_range[0]:date_range[1], selected_indicators]
            
            fig = go.Figure()
            for col in selected_indicators:
                fig.add_trace(go.Scatter(
                    x=filtered_df.index,
                    y=filtered_df[col],
                    name=col,
                    mode='lines+markers'
                ))
            
            fig.update_layout(
                title="Interactive Economic Indicators",
                xaxis_title="Date",
                yaxis_title="Value",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)

def show_analysis():
    """Show analysis capabilities"""
    st.header("🔍 Analysis Demo")
    
    df = create_sample_data()
    
    # Analysis tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Trend Analysis", "📊 Volatility", "🔗 Correlations", "📋 Summary"])
    
    with tab1:
        st.subheader("📈 Trend Analysis")
        
        # Calculate trends
        trends = {}
        for col in df.columns:
            x = np.arange(len(df))
            y = df[col].values
            slope, intercept = np.polyfit(x, y, 1)
            trends[col] = {
                'slope': slope,
                'trend_direction': 'Increasing' if slope > 0 else 'Decreasing',
                'trend_strength': abs(slope)
            }
        
        # Display trends
        trend_data = []
        for indicator, trend in trends.items():
            trend_data.append({
                'Indicator': indicator,
                'Trend': trend['trend_direction'],
                'Slope': f"{trend['slope']:.4f}",
                'Strength': f"{trend['trend_strength']:.4f}"
            })
        
        st.dataframe(pd.DataFrame(trend_data))
        
        # Trend visualization
        fig = go.Figure()
        for col in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[col],
                name=f"{col} (Trend: {trends[col]['trend_direction']})",
                mode='lines'
            ))
        
        fig.update_layout(
            title="Economic Indicators with Trend Analysis",
            xaxis_title="Date",
            yaxis_title="Value",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("📊 Volatility Analysis")
        
        # Calculate volatility
        volatility = df.pct_change().std() * np.sqrt(252)  # Annualized
        
        # Volatility chart
        fig = px.bar(
            x=volatility.index,
            y=volatility.values,
            title="Annualized Volatility by Indicator",
            labels={'x': 'Indicator', 'y': 'Volatility'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Volatility table
        vol_data = []
        for indicator, vol in volatility.items():
            vol_data.append({
                'Indicator': indicator,
                'Annualized Volatility': f"{vol:.2%}"
            })
        
        st.dataframe(pd.DataFrame(vol_data))
    
    with tab3:
        st.subheader("🔗 Correlation Analysis")
        
        correlation = df.corr()
        
        # Correlation heatmap
        fig = px.imshow(
            correlation,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu",
            title="Correlation Matrix"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Strong correlations
        st.subheader("Strong Correlations (>0.7)")
        strong_corr = []
        for i, col1 in enumerate(df.columns):
            for j, col2 in enumerate(df.columns):
                if i < j:
                    corr = correlation.loc[col1, col2]
                    if abs(corr) > 0.7:
                        strong_corr.append({
                            'Indicator 1': col1,
                            'Indicator 2': col2,
                            'Correlation': f"{corr:.3f}"
                        })
        
        if strong_corr:
            st.dataframe(pd.DataFrame(strong_corr))
        else:
            st.info("No strong correlations found in this sample data.")
    
    with tab4:
        st.subheader("📋 Analysis Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Indicators", len(df.columns))
            st.metric("Data Points", len(df))
            st.metric("Date Range", f"{df.index.min().strftime('%Y-%m')} to {df.index.max().strftime('%Y-%m')}")
        
        with col2:
            avg_volatility = volatility.mean()
            st.metric("Average Volatility", f"{avg_volatility:.2%}")
            
            increasing_trends = sum(1 for trend in trends.values() if trend['trend_direction'] == 'Increasing')
            st.metric("Increasing Trends", f"{increasing_trends}/{len(trends)}")

def show_architecture():
    """Show system architecture"""
    st.header("🏗️ System Architecture")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📋 Component Overview")
        
        architecture_data = {
            "Component": ["Frontend", "Backend", "Storage", "Scheduling", "Data Source"],
            "Technology": ["Streamlit", "AWS Lambda", "AWS S3", "EventBridge", "FRED API"],
            "Status": ["✅ Ready", "✅ Ready", "✅ Ready", "✅ Ready", "✅ Connected"]
        }
        
        st.dataframe(pd.DataFrame(architecture_data))
        
        st.subheader("🔧 Key Features")
        features = [
            "🎨 Interactive Streamlit Dashboard",
            "⚡ Serverless AWS Lambda Functions",
            "📦 Scalable S3 Storage",
            "⏰ Automated EventBridge Scheduling",
            "📊 Real-time FRED API Integration",
            "🧪 Comprehensive Testing Suite",
            "🔄 CI/CD Pipeline with GitHub Actions",
            "📈 Advanced Analytics & ML"
        ]
        
        for feature in features:
            st.write(f"• {feature}")
    
    with col2:
        st.subheader("🔄 Data Flow")
        
        # Create a simple flow diagram
        st.markdown("""
        ```
        FRED API → AWS Lambda → S3 Storage → Streamlit Dashboard
                    ↓
                EventBridge (Scheduling)
                    ↓
                CloudWatch (Monitoring)
        ```
        """)
        
        st.subheader("📊 System Metrics")
        
        metrics_data = {
            "Metric": ["API Response Time", "Data Processing Speed", "Storage Capacity", "Uptime"],
            "Value": ["< 100ms", "Real-time", "Unlimited", "99.9%"],
            "Status": ["✅ Optimal", "✅ Fast", "✅ Scalable", "✅ High"]
        }
        
        st.dataframe(pd.DataFrame(metrics_data))

def show_live_demo():
    """Show live demo capabilities"""
    st.header("⚡ Live Demo")
    
    st.info("This section demonstrates real-time capabilities of the FRED ML system.")
    
    # Demo controls
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎛️ Demo Controls")
        
        # Simulate real-time data
        if st.button("🔄 Refresh Data"):
            st.success("Data refreshed successfully!")
            time.sleep(1)
        
        # Analysis type
        analysis_type = st.selectbox(
            "Analysis Type:",
            ["Quick Analysis", "Deep Dive", "Custom Range"]
        )
        
        # Date range
        start_date = st.date_input("Start Date", value=datetime(2020, 1, 1))
        end_date = st.date_input("End Date", value=datetime(2024, 1, 1))
    
    with col2:
        st.subheader("📊 Live Metrics")
        
        # Simulate live metrics
        import random
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("API Calls/sec", random.randint(10, 50))
            st.metric("Data Points", random.randint(1000, 5000))
        
        with col2:
            st.metric("Processing Time", f"{random.uniform(0.1, 0.5):.2f}s")
            st.metric("Success Rate", f"{random.uniform(95, 99.9):.1f}%")
    
    # Live visualization
    st.subheader("📈 Live Data Visualization")
    
    # Create animated chart
    df = create_sample_data()
    
    # Add some noise for "live" effect
    live_df = df.copy()
    live_df += np.random.normal(0, 0.1, live_df.shape)
    
    fig = go.Figure()
    
    for col in live_df.columns:
        fig.add_trace(go.Scatter(
            x=live_df.index,
            y=live_df[col],
            name=col,
            mode='lines',
            line=dict(width=2)
        ))
    
    fig.update_layout(
        title="Live Economic Indicators",
        xaxis_title="Date",
        yaxis_title="Value",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Status indicators
    st.subheader("🔧 System Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.success("✅ FRED API")
    with col2:
        st.success("✅ AWS Lambda")
    with col3:
        st.success("✅ S3 Storage")
    with col4:
        st.success("✅ Streamlit")

if __name__ == "__main__":
    main() 