#!/usr/bin/env python3
"""
Comprehensive Economic Analytics Demo
Demonstrates advanced analytics capabilities including forecasting, segmentation, and statistical modeling
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.analysis.comprehensive_analytics import ComprehensiveAnalytics
from src.core.enhanced_fred_client import EnhancedFREDClient
from config.settings import FRED_API_KEY

def setup_logging():
    """Setup logging for demo"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def run_basic_demo():
    """Run basic demo with key economic indicators"""
    print("=" * 80)
    print("ECONOMIC ANALYTICS DEMO - BASIC ANALYSIS")
    print("=" * 80)
    
    # Initialize client
    client = EnhancedFREDClient(FRED_API_KEY)
    
    # Fetch data for key indicators
    indicators = ['GDPC1', 'INDPRO', 'RSAFS']
    print(f"\n📊 Fetching data for indicators: {indicators}")
    
    try:
        data = client.fetch_economic_data(
            indicators=indicators,
            start_date='2010-01-01',
            end_date='2024-01-01'
        )
        
        print(f"✅ Successfully fetched {len(data)} observations")
        print(f"📅 Date range: {data.index.min().strftime('%Y-%m')} to {data.index.max().strftime('%Y-%m')}")
        
        # Data quality report
        quality_report = client.validate_data_quality(data)
        print(f"\n📈 Data Quality Summary:")
        for series, metrics in quality_report['missing_data'].items():
            print(f"  • {series}: {metrics['completeness']:.1f}% complete")
        
        return data
        
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return None

def run_forecasting_demo(data):
    """Run forecasting demo"""
    print("\n" + "=" * 80)
    print("FORECASTING DEMO")
    print("=" * 80)
    
    from src.analysis.economic_forecasting import EconomicForecaster
    
    forecaster = EconomicForecaster(data)
    
    # Forecast key indicators
    indicators = ['GDPC1', 'INDPRO', 'RSAFS']
    available_indicators = [ind for ind in indicators if ind in data.columns]
    
    print(f"🔮 Forecasting indicators: {available_indicators}")
    
    for indicator in available_indicators:
        try:
            # Prepare data
            series = forecaster.prepare_data(indicator)
            
            # Check stationarity
            stationarity = forecaster.check_stationarity(series)
            print(f"\n📊 {indicator} Stationarity Test:")
            print(f"  • ADF Statistic: {stationarity['adf_statistic']:.4f}")
            print(f"  • P-value: {stationarity['p_value']:.4f}")
            print(f"  • Is Stationary: {stationarity['is_stationary']}")
            
            # Generate forecast
            forecast_result = forecaster.forecast_series(series, forecast_periods=4)
            print(f"🔮 {indicator} Forecast:")
            print(f"  • Model: {forecast_result['model_type'].upper()}")
            if forecast_result['aic']:
                print(f"  • AIC: {forecast_result['aic']:.4f}")
            
            # Backtest
            backtest_result = forecaster.backtest_forecast(series)
            if 'error' not in backtest_result:
                print(f"  • Backtest MAPE: {backtest_result['mape']:.2f}%")
                print(f"  • Backtest RMSE: {backtest_result['rmse']:.4f}")
            
        except Exception as e:
            print(f"❌ Error forecasting {indicator}: {e}")

def run_segmentation_demo(data):
    """Run segmentation demo"""
    print("\n" + "=" * 80)
    print("SEGMENTATION DEMO")
    print("=" * 80)
    
    from src.analysis.economic_segmentation import EconomicSegmentation
    
    segmentation = EconomicSegmentation(data)
    
    # Time period clustering
    print("🎯 Clustering time periods...")
    try:
        time_clusters = segmentation.cluster_time_periods(
            indicators=['GDPC1', 'INDPRO', 'RSAFS'],
            method='kmeans'
        )
        
        if 'error' not in time_clusters:
            n_clusters = time_clusters['n_clusters']
            print(f"✅ Time periods clustered into {n_clusters} economic regimes")
            
            # Show cluster analysis
            cluster_analysis = time_clusters['cluster_analysis']
            for cluster_id, analysis in cluster_analysis.items():
                print(f"  • Cluster {cluster_id}: {analysis['size']} periods ({analysis['percentage']:.1f}%)")
        
    except Exception as e:
        print(f"❌ Error in time period clustering: {e}")
    
    # Series clustering
    print("\n🎯 Clustering economic series...")
    try:
        series_clusters = segmentation.cluster_economic_series(
            indicators=['GDPC1', 'INDPRO', 'RSAFS', 'CPIAUCSL', 'FEDFUNDS', 'DGS10'],
            method='kmeans'
        )
        
        if 'error' not in series_clusters:
            n_clusters = series_clusters['n_clusters']
            print(f"✅ Economic series clustered into {n_clusters} groups")
            
            # Show cluster analysis
            cluster_analysis = series_clusters['cluster_analysis']
            for cluster_id, analysis in cluster_analysis.items():
                print(f"  • Cluster {cluster_id}: {analysis['size']} series ({analysis['percentage']:.1f}%)")
        
    except Exception as e:
        print(f"❌ Error in series clustering: {e}")

def run_statistical_demo(data):
    """Run statistical modeling demo"""
    print("\n" + "=" * 80)
    print("STATISTICAL MODELING DEMO")
    print("=" * 80)
    
    from src.analysis.statistical_modeling import StatisticalModeling
    
    modeling = StatisticalModeling(data)
    
    # Correlation analysis
    print("📊 Performing correlation analysis...")
    try:
        corr_results = modeling.analyze_correlations()
        significant_correlations = corr_results['significant_correlations']
        print(f"✅ Found {len(significant_correlations)} significant correlations")
        
        # Show top correlations
        print("\n🔗 Top 3 Strongest Correlations:")
        for i, corr in enumerate(significant_correlations[:3]):
            print(f"  • {corr['variable1']} ↔ {corr['variable2']}: {corr['correlation']:.3f} ({corr['strength']})")
        
    except Exception as e:
        print(f"❌ Error in correlation analysis: {e}")
    
    # Regression analysis
    print("\n📈 Performing regression analysis...")
    key_indicators = ['GDPC1', 'INDPRO', 'RSAFS']
    
    for target in key_indicators:
        if target in data.columns:
            try:
                regression_result = modeling.fit_regression_model(
                    target=target,
                    lag_periods=4
                )
                
                performance = regression_result['performance']
                print(f"✅ {target} Regression Model:")
                print(f"  • R²: {performance['r2']:.4f}")
                print(f"  • RMSE: {performance['rmse']:.4f}")
                print(f"  • MAE: {performance['mae']:.4f}")
                
                # Show top coefficients
                coefficients = regression_result['coefficients']
                print(f"  • Top 3 Variables:")
                for i, row in coefficients.head(3).iterrows():
                    print(f"    - {row['variable']}: {row['coefficient']:.4f}")
                
            except Exception as e:
                print(f"❌ Error in regression for {target}: {e}")

def run_comprehensive_demo():
    """Run comprehensive analytics demo"""
    print("=" * 80)
    print("COMPREHENSIVE ECONOMIC ANALYTICS DEMO")
    print("=" * 80)
    
    # Initialize comprehensive analytics
    analytics = ComprehensiveAnalytics(FRED_API_KEY, output_dir="data/exports/demo")
    
    # Run complete analysis
    print("\n🚀 Running comprehensive analysis...")
    try:
        results = analytics.run_complete_analysis(
            indicators=['GDPC1', 'INDPRO', 'RSAFS', 'CPIAUCSL', 'FEDFUNDS', 'DGS10'],
            start_date='2010-01-01',
            end_date='2024-01-01',
            forecast_periods=4,
            include_visualizations=True
        )
        
        print("✅ Comprehensive analysis completed successfully!")
        
        # Print key insights
        if 'insights' in results:
            insights = results['insights']
            print("\n🎯 KEY INSIGHTS:")
            for finding in insights.get('key_findings', []):
                print(f"  • {finding}")
        
        # Print forecasting results
        if 'forecasting' in results:
            print("\n🔮 FORECASTING RESULTS:")
            forecasting_results = results['forecasting']
            for indicator, result in forecasting_results.items():
                if 'error' not in result:
                    backtest = result.get('backtest', {})
                    if 'error' not in backtest:
                        mape = backtest.get('mape', 0)
                        print(f"  • {indicator}: MAPE = {mape:.2f}%")
        
        # Print segmentation results
        if 'segmentation' in results:
            print("\n🎯 SEGMENTATION RESULTS:")
            segmentation_results = results['segmentation']
            
            if 'time_period_clusters' in segmentation_results:
                time_clusters = segmentation_results['time_period_clusters']
                if 'error' not in time_clusters:
                    n_clusters = time_clusters.get('n_clusters', 0)
                    print(f"  • Time periods clustered into {n_clusters} economic regimes")
            
            if 'series_clusters' in segmentation_results:
                series_clusters = segmentation_results['series_clusters']
                if 'error' not in series_clusters:
                    n_clusters = series_clusters.get('n_clusters', 0)
                    print(f"  • Economic series clustered into {n_clusters} groups")
        
        print(f"\n📁 Results saved to: data/exports/demo")
        
    except Exception as e:
        print(f"❌ Error in comprehensive analysis: {e}")

def main():
    """Main demo function"""
    setup_logging()
    
    print("🎯 ECONOMIC ANALYTICS DEMO")
    print("This demo showcases advanced analytics capabilities including:")
    print("  • Economic data collection and quality assessment")
    print("  • Time series forecasting with ARIMA/ETS models")
    print("  • Economic segmentation (time periods and series)")
    print("  • Statistical modeling and correlation analysis")
    print("  • Comprehensive insights extraction")
    
    # Check if API key is available
    if not FRED_API_KEY:
        print("\n❌ FRED API key not found. Please set FRED_API_KEY environment variable.")
        return
    
    # Run basic demo
    data = run_basic_demo()
    if data is None:
        return
    
    # Run individual demos
    run_forecasting_demo(data)
    run_segmentation_demo(data)
    run_statistical_demo(data)
    
    # Run comprehensive demo
    run_comprehensive_demo()
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETED!")
    print("=" * 80)
    print("Generated outputs:")
    print("  📊 data/exports/demo/ - Comprehensive analysis results")
    print("  📈 Visualizations and reports")
    print("  📉 Statistical diagnostics")
    print("  🔮 Forecasting results")
    print("  🎯 Segmentation analysis")

if __name__ == "__main__":
    main() 