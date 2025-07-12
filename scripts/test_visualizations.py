#!/usr/bin/env python3
"""
Test script for visualization generation and S3 storage
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.visualization.chart_generator import ChartGenerator

def test_visualization_generation():
    """Test the visualization generation functionality"""
    print("🧪 Testing visualization generation...")
    
    try:
        # Create sample economic data
        dates = pd.date_range('2020-01-01', periods=50, freq='M')
        sample_data = pd.DataFrame({
            'GDPC1': np.random.normal(100, 10, 50),
            'INDPRO': np.random.normal(50, 5, 50),
            'CPIAUCSL': np.random.normal(200, 20, 50),
            'FEDFUNDS': np.random.normal(2, 0.5, 50),
            'UNRATE': np.random.normal(4, 1, 50)
        }, index=dates)
        
        print(f"✅ Created sample data with shape: {sample_data.shape}")
        
        # Initialize chart generator
        chart_gen = ChartGenerator()
        print("✅ Initialized ChartGenerator")
        
        # Test individual chart generation
        print("\n📊 Testing individual chart generation...")
        
        # Time series chart
        time_series_key = chart_gen.create_time_series_chart(sample_data)
        if time_series_key:
            print(f"✅ Time series chart created: {time_series_key}")
        else:
            print("❌ Time series chart failed")
        
        # Correlation heatmap
        correlation_key = chart_gen.create_correlation_heatmap(sample_data)
        if correlation_key:
            print(f"✅ Correlation heatmap created: {correlation_key}")
        else:
            print("❌ Correlation heatmap failed")
        
        # Distribution charts
        distribution_keys = chart_gen.create_distribution_charts(sample_data)
        if distribution_keys:
            print(f"✅ Distribution charts created: {len(distribution_keys)} charts")
        else:
            print("❌ Distribution charts failed")
        
        # PCA visualization
        pca_key = chart_gen.create_pca_visualization(sample_data)
        if pca_key:
            print(f"✅ PCA visualization created: {pca_key}")
        else:
            print("❌ PCA visualization failed")
        
        # Clustering chart
        clustering_key = chart_gen.create_clustering_chart(sample_data)
        if clustering_key:
            print(f"✅ Clustering chart created: {clustering_key}")
        else:
            print("❌ Clustering chart failed")
        
        # Test comprehensive visualization generation
        print("\n🎯 Testing comprehensive visualization generation...")
        visualizations = chart_gen.generate_comprehensive_visualizations(sample_data, "comprehensive")
        
        if visualizations:
            print(f"✅ Generated {len(visualizations)} comprehensive visualizations:")
            for chart_type, chart_key in visualizations.items():
                print(f"  - {chart_type}: {chart_key}")
        else:
            print("❌ Comprehensive visualization generation failed")
        
        # Test chart listing
        print("\n📋 Testing chart listing...")
        charts = chart_gen.list_available_charts()
        if charts:
            print(f"✅ Found {len(charts)} charts in S3")
            for chart in charts[:3]:  # Show first 3
                print(f"  - {chart['key']} ({chart['size']} bytes)")
        else:
            print("ℹ️ No charts found in S3 (this is normal for first run)")
        
        print("\n🎉 Visualization tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Visualization test failed: {e}")
        return False

def test_chart_retrieval():
    """Test retrieving charts from S3"""
    print("\n🔄 Testing chart retrieval...")
    
    try:
        chart_gen = ChartGenerator()
        charts = chart_gen.list_available_charts()
        
        if charts:
            # Test retrieving the first chart
            first_chart = charts[0]
            print(f"Testing retrieval of: {first_chart['key']}")
            
            response = chart_gen.s3_client.get_object(
                Bucket=chart_gen.s3_bucket,
                Key=first_chart['key']
            )
            chart_data = response['Body'].read()
            
            print(f"✅ Successfully retrieved chart ({len(chart_data)} bytes)")
            return True
        else:
            print("ℹ️ No charts available for retrieval test")
            return True
            
    except Exception as e:
        print(f"❌ Chart retrieval test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting visualization tests...")
    
    # Test visualization generation
    gen_success = test_visualization_generation()
    
    # Test chart retrieval
    retrieval_success = test_chart_retrieval()
    
    if gen_success and retrieval_success:
        print("\n✅ All visualization tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some visualization tests failed!")
        sys.exit(1) 