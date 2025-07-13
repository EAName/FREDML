#!/usr/bin/env python3
"""
Advanced Analytics Runner
Executes comprehensive economic analytics pipeline with forecasting, segmentation, and statistical modeling
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.analysis.comprehensive_analytics import ComprehensiveAnalytics
from config.settings import FRED_API_KEY

def setup_logging(log_level: str = 'INFO'):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/advanced_analytics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )

def main():
    """Main function to run advanced analytics pipeline"""
    parser = argparse.ArgumentParser(description='Run comprehensive economic analytics pipeline')
    parser.add_argument('--api-key', type=str, help='FRED API key (overrides config)')
    parser.add_argument('--indicators', nargs='+', 
                       default=['GDPC1', 'INDPRO', 'RSAFS', 'CPIAUCSL', 'FEDFUNDS', 'DGS10'],
                       help='Economic indicators to analyze')
    parser.add_argument('--start-date', type=str, default='1990-01-01',
                       help='Start date for analysis (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                       help='End date for analysis (YYYY-MM-DD)')
    parser.add_argument('--forecast-periods', type=int, default=4,
                       help='Number of periods to forecast')
    parser.add_argument('--output-dir', type=str, default='data/exports',
                       help='Output directory for results')
    parser.add_argument('--no-visualizations', action='store_true',
                       help='Skip visualization generation')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Create logs directory
    Path('logs').mkdir(exist_ok=True)
    
    # Get API key
    api_key = args.api_key or FRED_API_KEY
    if not api_key:
        logger.error("FRED API key not provided. Set FRED_API_KEY environment variable or use --api-key")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting Advanced Economic Analytics Pipeline")
    logger.info(f"Indicators: {args.indicators}")
    logger.info(f"Date range: {args.start_date} to {args.end_date or 'current'}")
    logger.info(f"Forecast periods: {args.forecast_periods}")
    logger.info(f"Output directory: {output_dir}")
    
    try:
        # Initialize analytics pipeline
        analytics = ComprehensiveAnalytics(api_key=api_key, output_dir=str(output_dir))
        
        # Run complete analysis
        results = analytics.run_complete_analysis(
            indicators=args.indicators,
            start_date=args.start_date,
            end_date=args.end_date,
            forecast_periods=args.forecast_periods,
            include_visualizations=not args.no_visualizations
        )
        
        # Print summary
        logger.info("Analysis completed successfully!")
        logger.info(f"Results saved to: {output_dir}")
        
        # Print key insights
        if 'insights' in results:
            insights = results['insights']
            logger.info("\nKEY INSIGHTS:")
            for finding in insights.get('key_findings', []):
                logger.info(f"  • {finding}")
            
            # Print top insights by category
            for insight_type, insight_list in insights.items():
                if insight_type != 'key_findings' and insight_list:
                    logger.info(f"\n{insight_type.replace('_', ' ').title()}:")
                    for insight in insight_list[:3]:  # Top 3 insights
                        logger.info(f"  • {insight}")
        
        # Print forecasting results
        if 'forecasting' in results:
            logger.info("\nFORECASTING RESULTS:")
            forecasting_results = results['forecasting']
            for indicator, result in forecasting_results.items():
                if 'error' not in result:
                    backtest = result.get('backtest', {})
                    if 'error' not in backtest:
                        mape = backtest.get('mape', 0)
                        logger.info(f"  • {indicator}: MAPE = {mape:.2f}%")
        
        # Print segmentation results
        if 'segmentation' in results:
            logger.info("\nSEGMENTATION RESULTS:")
            segmentation_results = results['segmentation']
            
            if 'time_period_clusters' in segmentation_results:
                time_clusters = segmentation_results['time_period_clusters']
                if 'error' not in time_clusters:
                    n_clusters = time_clusters.get('n_clusters', 0)
                    logger.info(f"  • Time periods clustered into {n_clusters} economic regimes")
            
            if 'series_clusters' in segmentation_results:
                series_clusters = segmentation_results['series_clusters']
                if 'error' not in series_clusters:
                    n_clusters = series_clusters.get('n_clusters', 0)
                    logger.info(f"  • Economic series clustered into {n_clusters} groups")
        
        # Print statistical results
        if 'statistical_modeling' in results:
            logger.info("\nSTATISTICAL ANALYSIS RESULTS:")
            stat_results = results['statistical_modeling']
            
            if 'correlation' in stat_results:
                corr_results = stat_results['correlation']
                significant_correlations = corr_results.get('significant_correlations', [])
                logger.info(f"  • {len(significant_correlations)} significant correlations identified")
            
            if 'regression' in stat_results:
                reg_results = stat_results['regression']
                successful_models = [k for k, v in reg_results.items() if 'error' not in v]
                logger.info(f"  • {len(successful_models)} regression models successfully fitted")
        
        logger.info(f"\nDetailed reports and visualizations saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)

if __name__ == "__main__":
    main() 