#!/usr/bin/env python
"""
Advanced Analytics Runner for FRED Economic Data
Runs comprehensive statistical analysis, modeling, and insights extraction.
"""

import os
import sys
import glob
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from analysis.advanced_analytics import AdvancedAnalytics

def find_latest_data():
    """Find the most recent FRED data file."""
    data_files = glob.glob('data/processed/fred_data_*.csv')
    if not data_files:
        raise FileNotFoundError("No FRED data files found. Run the pipeline first.")
    
    # Get the most recent file
    latest_file = max(data_files, key=os.path.getctime)
    print(f"Using data file: {latest_file}")
    return latest_file

def main():
    """Run the complete advanced analytics workflow."""
    print("=" * 80)
    print("FRED ECONOMIC DATA - ADVANCED ANALYTICS")
    print("=" * 80)
    
    try:
        # Find the latest data file
        data_file = find_latest_data()
        
        # Initialize analytics
        analytics = AdvancedAnalytics(data_path=data_file)
        
        # Run complete analysis
        results = analytics.run_complete_analysis()
        
        print("\n" + "=" * 80)
        print("ANALYTICS COMPLETE!")
        print("=" * 80)
        print("Generated outputs:")
        print("  📊 data/exports/insights_report.txt - Comprehensive insights")
        print("  📈 data/exports/clustering_analysis.png - Clustering results")
        print("  📉 data/exports/time_series_decomposition.png - Time series decomposition")
        print("  🔮 data/exports/time_series_forecast.png - Time series forecast")
        print("\nKey findings have been saved to data/exports/insights_report.txt")
        
    except Exception as e:
        print(f"Error running analytics: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 