"""
Comprehensive Analytics Pipeline
Orchestrates advanced analytics including forecasting, segmentation, statistical modeling, and insights
"""

import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

from src.analysis.economic_forecasting import EconomicForecaster
from src.analysis.economic_segmentation import EconomicSegmentation
from src.analysis.statistical_modeling import StatisticalModeling
from src.core.enhanced_fred_client import EnhancedFREDClient

logger = logging.getLogger(__name__)

class ComprehensiveAnalytics:
    """
    Comprehensive analytics pipeline for economic data analysis
    combining forecasting, segmentation, statistical modeling, and insights extraction
    """
    
    def __init__(self, api_key: str, output_dir: str = "data/exports"):
        """
        Initialize comprehensive analytics pipeline
        
        Args:
            api_key: FRED API key
            output_dir: Output directory for results
        """
        self.client = EnhancedFREDClient(api_key)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize analytics modules
        self.forecaster = None
        self.segmentation = None
        self.statistical_modeling = None
        
        # Results storage
        self.data = None
        self.results = {}
        self.reports = {}
        
    def run_complete_analysis(self, indicators: List[str] = None,
                            start_date: str = '1990-01-01',
                            end_date: str = None,
                            forecast_periods: int = 4,
                            include_visualizations: bool = True) -> Dict:
        """
        Run complete advanced analytics pipeline
        
        Args:
            indicators: List of economic indicators to analyze
            start_date: Start date for analysis
            end_date: End date for analysis
            forecast_periods: Number of periods to forecast
            include_visualizations: Whether to generate visualizations
            
        Returns:
            Dictionary with all analysis results
        """
        logger.info("Starting comprehensive economic analytics pipeline")
        
        # Step 1: Data Collection
        logger.info("Step 1: Collecting economic data")
        self.data = self.client.fetch_economic_data(
            indicators=indicators,
            start_date=start_date,
            end_date=end_date,
            frequency='auto'
        )
        
        # Step 2: Data Quality Assessment
        logger.info("Step 2: Assessing data quality")
        quality_report = self.client.validate_data_quality(self.data)
        self.results['data_quality'] = quality_report
        
        # Step 3: Initialize Analytics Modules
        logger.info("Step 3: Initializing analytics modules")
        self.forecaster = EconomicForecaster(self.data)
        self.segmentation = EconomicSegmentation(self.data)
        self.statistical_modeling = StatisticalModeling(self.data)
        
        # Step 4: Statistical Modeling
        logger.info("Step 4: Performing statistical modeling")
        statistical_results = self._run_statistical_analysis()
        self.results['statistical_modeling'] = statistical_results
        
        # Step 5: Economic Forecasting
        logger.info("Step 5: Performing economic forecasting")
        forecasting_results = self._run_forecasting_analysis(forecast_periods)
        self.results['forecasting'] = forecasting_results
        
        # Step 6: Economic Segmentation
        logger.info("Step 6: Performing economic segmentation")
        segmentation_results = self._run_segmentation_analysis()
        self.results['segmentation'] = segmentation_results
        
        # Step 7: Insights Extraction
        logger.info("Step 7: Extracting insights")
        insights = self._extract_insights()
        self.results['insights'] = insights
        
        # Step 8: Generate Reports and Visualizations
        logger.info("Step 8: Generating reports and visualizations")
        if include_visualizations:
            self._generate_visualizations()
        
        self._generate_comprehensive_report()
        
        logger.info("Comprehensive analytics pipeline completed successfully")
        return self.results
    
    def _run_statistical_analysis(self) -> Dict:
        """Run comprehensive statistical analysis"""
        results = {}
        
        # Correlation analysis
        logger.info("  - Performing correlation analysis")
        correlation_results = self.statistical_modeling.analyze_correlations()
        results['correlation'] = correlation_results
        
        # Regression analysis for key indicators
        key_indicators = ['GDPC1', 'INDPRO', 'RSAFS']
        regression_results = {}
        
        for target in key_indicators:
            if target in self.data.columns:
                logger.info(f"  - Fitting regression model for {target}")
                try:
                    regression_result = self.statistical_modeling.fit_regression_model(
                        target=target,
                        lag_periods=4,
                        include_interactions=False
                    )
                    regression_results[target] = regression_result
                except Exception as e:
                    logger.warning(f"Regression failed for {target}: {e}")
                    regression_results[target] = {'error': str(e)}
        
        results['regression'] = regression_results
        
        # Granger causality analysis
        logger.info("  - Performing Granger causality analysis")
        causality_results = {}
        for target in key_indicators:
            if target in self.data.columns:
                causality_results[target] = {}
                for predictor in self.data.columns:
                    if predictor != target:
                        try:
                            causality_result = self.statistical_modeling.perform_granger_causality(
                                target=target,
                                predictor=predictor,
                                max_lags=4
                            )
                            causality_results[target][predictor] = causality_result
                        except Exception as e:
                            logger.warning(f"Causality test failed for {target} -> {predictor}: {e}")
                            causality_results[target][predictor] = {'error': str(e)}
        
        results['causality'] = causality_results
        
        return results
    
    def _run_forecasting_analysis(self, forecast_periods: int) -> Dict:
        """Run comprehensive forecasting analysis"""
        logger.info("  - Forecasting economic indicators")
        
        # Focus on key indicators for forecasting
        key_indicators = ['GDPC1', 'INDPRO', 'RSAFS']
        available_indicators = [ind for ind in key_indicators if ind in self.data.columns]
        
        if not available_indicators:
            logger.warning("No key indicators available for forecasting")
            return {'error': 'No suitable indicators for forecasting'}
        
        # Perform forecasting
        forecasting_results = self.forecaster.forecast_economic_indicators(available_indicators)
        
        return forecasting_results
    
    def _run_segmentation_analysis(self) -> Dict:
        """Run comprehensive segmentation analysis"""
        results = {}
        
        # Time period clustering
        logger.info("  - Clustering time periods")
        try:
            time_period_clusters = self.segmentation.cluster_time_periods(
                indicators=['GDPC1', 'INDPRO', 'RSAFS'],
                method='kmeans'
            )
            results['time_period_clusters'] = time_period_clusters
        except Exception as e:
            logger.warning(f"Time period clustering failed: {e}")
            results['time_period_clusters'] = {'error': str(e)}
        
        # Series clustering
        logger.info("  - Clustering economic series")
        try:
            series_clusters = self.segmentation.cluster_economic_series(
                indicators=['GDPC1', 'INDPRO', 'RSAFS', 'CPIAUCSL', 'FEDFUNDS', 'DGS10'],
                method='kmeans'
            )
            results['series_clusters'] = series_clusters
        except Exception as e:
            logger.warning(f"Series clustering failed: {e}")
            results['series_clusters'] = {'error': str(e)}
        
        return results
    
    def _extract_insights(self) -> Dict:
        """Extract key insights from all analyses"""
        insights = {
            'key_findings': [],
            'economic_indicators': {},
            'forecasting_insights': [],
            'segmentation_insights': [],
            'statistical_insights': []
        }
        
        # Extract insights from forecasting
        if 'forecasting' in self.results:
            forecasting_results = self.results['forecasting']
            for indicator, result in forecasting_results.items():
                if 'error' not in result:
                    # Model performance insights
                    backtest = result.get('backtest', {})
                    if 'error' not in backtest:
                        mape = backtest.get('mape', 0)
                        if mape < 5:
                            insights['forecasting_insights'].append(
                                f"{indicator} forecasting shows excellent accuracy (MAPE: {mape:.2f}%)"
                            )
                        elif mape < 10:
                            insights['forecasting_insights'].append(
                                f"{indicator} forecasting shows good accuracy (MAPE: {mape:.2f}%)"
                            )
                        else:
                            insights['forecasting_insights'].append(
                                f"{indicator} forecasting shows moderate accuracy (MAPE: {mape:.2f}%)"
                            )
                    
                    # Stationarity insights
                    stationarity = result.get('stationarity', {})
                    if 'is_stationary' in stationarity:
                        if stationarity['is_stationary']:
                            insights['forecasting_insights'].append(
                                f"{indicator} series is stationary, suitable for time series modeling"
                            )
                        else:
                            insights['forecasting_insights'].append(
                                f"{indicator} series is non-stationary, may require differencing"
                            )
        
        # Extract insights from segmentation
        if 'segmentation' in self.results:
            segmentation_results = self.results['segmentation']
            
            # Time period clustering insights
            if 'time_period_clusters' in segmentation_results:
                time_clusters = segmentation_results['time_period_clusters']
                if 'error' not in time_clusters:
                    n_clusters = time_clusters.get('n_clusters', 0)
                    insights['segmentation_insights'].append(
                        f"Time periods clustered into {n_clusters} distinct economic regimes"
                    )
            
            # Series clustering insights
            if 'series_clusters' in segmentation_results:
                series_clusters = segmentation_results['series_clusters']
                if 'error' not in series_clusters:
                    n_clusters = series_clusters.get('n_clusters', 0)
                    insights['segmentation_insights'].append(
                        f"Economic series clustered into {n_clusters} groups based on behavior patterns"
                    )
        
        # Extract insights from statistical modeling
        if 'statistical_modeling' in self.results:
            stat_results = self.results['statistical_modeling']
            
            # Correlation insights
            if 'correlation' in stat_results:
                corr_results = stat_results['correlation']
                significant_correlations = corr_results.get('significant_correlations', [])
                
                if significant_correlations:
                    strongest_corr = significant_correlations[0]
                    insights['statistical_insights'].append(
                        f"Strongest correlation: {strongest_corr['variable1']} ↔ {strongest_corr['variable2']} "
                        f"(r={strongest_corr['correlation']:.3f})"
                    )
            
            # Regression insights
            if 'regression' in stat_results:
                reg_results = stat_results['regression']
                for target, result in reg_results.items():
                    if 'error' not in result:
                        performance = result.get('performance', {})
                        r2 = performance.get('r2', 0)
                        if r2 > 0.7:
                            insights['statistical_insights'].append(
                                f"{target} regression model shows strong explanatory power (R² = {r2:.3f})"
                            )
                        elif r2 > 0.5:
                            insights['statistical_insights'].append(
                                f"{target} regression model shows moderate explanatory power (R² = {r2:.3f})"
                            )
        
        # Generate key findings
        insights['key_findings'] = [
            f"Analysis covers {len(self.data.columns)} economic indicators from {self.data.index.min().strftime('%Y-%m')} to {self.data.index.max().strftime('%Y-%m')}",
            f"Dataset contains {len(self.data)} observations with {self.data.shape[0] * self.data.shape[1]} total data points",
            f"Generated {len(insights['forecasting_insights'])} forecasting insights",
            f"Generated {len(insights['segmentation_insights'])} segmentation insights",
            f"Generated {len(insights['statistical_insights'])} statistical insights"
        ]
        
        return insights
    
    def _generate_visualizations(self):
        """Generate comprehensive visualizations"""
        logger.info("Generating visualizations")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Time Series Plot
        self._plot_time_series()
        
        # 2. Correlation Heatmap
        self._plot_correlation_heatmap()
        
        # 3. Forecasting Results
        self._plot_forecasting_results()
        
        # 4. Segmentation Results
        self._plot_segmentation_results()
        
        # 5. Statistical Diagnostics
        self._plot_statistical_diagnostics()
        
        logger.info("Visualizations generated successfully")
    
    def _plot_time_series(self):
        """Plot time series of economic indicators"""
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        key_indicators = ['GDPC1', 'INDPRO', 'RSAFS', 'CPIAUCSL', 'FEDFUNDS', 'DGS10']
        
        for i, indicator in enumerate(key_indicators):
            if indicator in self.data.columns and i < len(axes):
                series = self.data[indicator].dropna()
                axes[i].plot(series.index, series.values, linewidth=1.5)
                axes[i].set_title(f'{indicator} - {self.client.ECONOMIC_INDICATORS.get(indicator, indicator)}')
                axes[i].set_xlabel('Date')
                axes[i].set_ylabel('Value')
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'economic_indicators_time_series.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_correlation_heatmap(self):
        """Plot correlation heatmap"""
        if 'statistical_modeling' in self.results:
            corr_results = self.results['statistical_modeling'].get('correlation', {})
            if 'correlation_matrix' in corr_results:
                corr_matrix = corr_results['correlation_matrix']
                
                plt.figure(figsize=(12, 10))
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                           square=True, linewidths=0.5, cbar_kws={"shrink": .8})
                plt.title('Economic Indicators Correlation Matrix')
                plt.tight_layout()
                plt.savefig(self.output_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
                plt.close()
    
    def _plot_forecasting_results(self):
        """Plot forecasting results"""
        if 'forecasting' in self.results:
            forecasting_results = self.results['forecasting']
            
            n_indicators = len([k for k, v in forecasting_results.items() if 'error' not in v])
            if n_indicators > 0:
                fig, axes = plt.subplots(n_indicators, 1, figsize=(15, 5*n_indicators))
                if n_indicators == 1:
                    axes = [axes]
                
                i = 0
                for indicator, result in forecasting_results.items():
                    if 'error' not in result and i < len(axes):
                        series = result.get('series', pd.Series())
                        forecast = result.get('forecast', {})
                        
                        if not series.empty and 'forecast' in forecast:
                            # Plot historical data
                            axes[i].plot(series.index, series.values, label='Historical', linewidth=2)
                            
                            # Plot forecast
                            if hasattr(forecast['forecast'], 'index'):
                                forecast_values = forecast['forecast']
                                forecast_index = pd.date_range(
                                    start=series.index[-1] + pd.DateOffset(months=3),
                                    periods=len(forecast_values),
                                    freq='Q'
                                )
                                axes[i].plot(forecast_index, forecast_values, 'r--', 
                                           label='Forecast', linewidth=2)
                            
                            axes[i].set_title(f'{indicator} - Forecast')
                            axes[i].set_xlabel('Date')
                            axes[i].set_ylabel('Growth Rate')
                            axes[i].legend()
                            axes[i].grid(True, alpha=0.3)
                            i += 1
                
                plt.tight_layout()
                plt.savefig(self.output_dir / 'forecasting_results.png', dpi=300, bbox_inches='tight')
                plt.close()
    
    def _plot_segmentation_results(self):
        """Plot segmentation results"""
        if 'segmentation' in self.results:
            segmentation_results = self.results['segmentation']
            
            # Plot time period clusters
            if 'time_period_clusters' in segmentation_results:
                time_clusters = segmentation_results['time_period_clusters']
                if 'error' not in time_clusters and 'pca_data' in time_clusters:
                    pca_data = time_clusters['pca_data']
                    cluster_labels = time_clusters['cluster_labels']
                    
                    plt.figure(figsize=(10, 8))
                    scatter = plt.scatter(pca_data[:, 0], pca_data[:, 1], 
                                       c=cluster_labels, cmap='viridis', alpha=0.7)
                    plt.colorbar(scatter)
                    plt.title('Time Period Clustering (PCA)')
                    plt.xlabel('Principal Component 1')
                    plt.ylabel('Principal Component 2')
                    plt.tight_layout()
                    plt.savefig(self.output_dir / 'time_period_clustering.png', dpi=300, bbox_inches='tight')
                    plt.close()
    
    def _plot_statistical_diagnostics(self):
        """Plot statistical diagnostics"""
        if 'statistical_modeling' in self.results:
            stat_results = self.results['statistical_modeling']
            
            # Plot regression diagnostics
            if 'regression' in stat_results:
                reg_results = stat_results['regression']
                
                for target, result in reg_results.items():
                    if 'error' not in result and 'residuals' in result:
                        residuals = result['residuals']
                        
                        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                        
                        # Residuals vs fitted
                        predictions = result.get('predictions', [])
                        if len(predictions) == len(residuals):
                            axes[0, 0].scatter(predictions, residuals, alpha=0.6)
                            axes[0, 0].axhline(y=0, color='r', linestyle='--')
                            axes[0, 0].set_title('Residuals vs Fitted')
                            axes[0, 0].set_xlabel('Fitted Values')
                            axes[0, 0].set_ylabel('Residuals')
                        
                        # Q-Q plot
                        from scipy import stats
                        stats.probplot(residuals, dist="norm", plot=axes[0, 1])
                        axes[0, 1].set_title('Q-Q Plot')
                        
                        # Histogram of residuals
                        axes[1, 0].hist(residuals, bins=20, alpha=0.7, edgecolor='black')
                        axes[1, 0].set_title('Residuals Distribution')
                        axes[1, 0].set_xlabel('Residuals')
                        axes[1, 0].set_ylabel('Frequency')
                        
                        # Time series of residuals
                        axes[1, 1].plot(residuals.index, residuals.values)
                        axes[1, 1].axhline(y=0, color='r', linestyle='--')
                        axes[1, 1].set_title('Residuals Time Series')
                        axes[1, 1].set_xlabel('Time')
                        axes[1, 1].set_ylabel('Residuals')
                        
                        plt.suptitle(f'Regression Diagnostics - {target}')
                        plt.tight_layout()
                        plt.savefig(self.output_dir / f'regression_diagnostics_{target}.png', 
                                  dpi=300, bbox_inches='tight')
                        plt.close()
    
    def _generate_comprehensive_report(self):
        """Generate comprehensive analysis report"""
        logger.info("Generating comprehensive report")
        
        # Generate individual reports
        if 'statistical_modeling' in self.results:
            stat_report = self.statistical_modeling.generate_statistical_report(
                regression_results=self.results['statistical_modeling'].get('regression'),
                correlation_results=self.results['statistical_modeling'].get('correlation'),
                causality_results=self.results['statistical_modeling'].get('causality')
            )
            self.reports['statistical'] = stat_report
        
        if 'forecasting' in self.results:
            forecast_report = self.forecaster.generate_forecast_report(self.results['forecasting'])
            self.reports['forecasting'] = forecast_report
        
        if 'segmentation' in self.results:
            segmentation_report = self.segmentation.generate_segmentation_report(
                time_period_clusters=self.results['segmentation'].get('time_period_clusters'),
                series_clusters=self.results['segmentation'].get('series_clusters')
            )
            self.reports['segmentation'] = segmentation_report
        
        # Generate comprehensive report
        comprehensive_report = self._generate_comprehensive_summary()
        
        # Save reports
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        with open(self.output_dir / f'comprehensive_analysis_report_{timestamp}.txt', 'w') as f:
            f.write(comprehensive_report)
        
        # Save individual reports
        for report_name, report_content in self.reports.items():
            with open(self.output_dir / f'{report_name}_report_{timestamp}.txt', 'w') as f:
                f.write(report_content)
        
        logger.info(f"Reports saved to {self.output_dir}")
    
    def _generate_comprehensive_summary(self) -> str:
        """Generate comprehensive summary report"""
        summary = "COMPREHENSIVE ECONOMIC ANALYTICS REPORT\n"
        summary += "=" * 60 + "\n\n"
        
        # Executive Summary
        summary += "EXECUTIVE SUMMARY\n"
        summary += "-" * 30 + "\n"
        
        if 'insights' in self.results:
            insights = self.results['insights']
            summary += f"Key Findings:\n"
            for finding in insights.get('key_findings', []):
                summary += f"  • {finding}\n"
            summary += "\n"
        
        # Data Overview
        summary += "DATA OVERVIEW\n"
        summary += "-" * 30 + "\n"
        summary += self.client.generate_data_summary(self.data)
        
        # Analysis Results Summary
        summary += "ANALYSIS RESULTS SUMMARY\n"
        summary += "-" * 30 + "\n"
        
        # Forecasting Summary
        if 'forecasting' in self.results:
            summary += "Forecasting Results:\n"
            forecasting_results = self.results['forecasting']
            for indicator, result in forecasting_results.items():
                if 'error' not in result:
                    backtest = result.get('backtest', {})
                    if 'error' not in backtest:
                        mape = backtest.get('mape', 0)
                        summary += f"  • {indicator}: MAPE = {mape:.2f}%\n"
            summary += "\n"
        
        # Segmentation Summary
        if 'segmentation' in self.results:
            summary += "Segmentation Results:\n"
            segmentation_results = self.results['segmentation']
            
            if 'time_period_clusters' in segmentation_results:
                time_clusters = segmentation_results['time_period_clusters']
                if 'error' not in time_clusters:
                    n_clusters = time_clusters.get('n_clusters', 0)
                    summary += f"  • Time periods clustered into {n_clusters} economic regimes\n"
            
            if 'series_clusters' in segmentation_results:
                series_clusters = segmentation_results['series_clusters']
                if 'error' not in series_clusters:
                    n_clusters = series_clusters.get('n_clusters', 0)
                    summary += f"  • Economic series clustered into {n_clusters} groups\n"
            summary += "\n"
        
        # Statistical Summary
        if 'statistical_modeling' in self.results:
            summary += "Statistical Analysis Results:\n"
            stat_results = self.results['statistical_modeling']
            
            if 'correlation' in stat_results:
                corr_results = stat_results['correlation']
                significant_correlations = corr_results.get('significant_correlations', [])
                summary += f"  • {len(significant_correlations)} significant correlations identified\n"
            
            if 'regression' in stat_results:
                reg_results = stat_results['regression']
                successful_models = [k for k, v in reg_results.items() if 'error' not in v]
                summary += f"  • {len(successful_models)} regression models successfully fitted\n"
            summary += "\n"
        
        # Key Insights
        if 'insights' in self.results:
            insights = self.results['insights']
            summary += "KEY INSIGHTS\n"
            summary += "-" * 30 + "\n"
            
            for insight_type, insight_list in insights.items():
                if insight_type != 'key_findings' and insight_list:
                    summary += f"{insight_type.replace('_', ' ').title()}:\n"
                    for insight in insight_list[:3]:  # Top 3 insights
                        summary += f"  • {insight}\n"
                    summary += "\n"
        
        summary += "=" * 60 + "\n"
        summary += f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        summary += f"Analysis period: {self.data.index.min().strftime('%Y-%m')} to {self.data.index.max().strftime('%Y-%m')}\n"
        
        return summary 