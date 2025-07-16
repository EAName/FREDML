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

# Optional imports with error handling
try:
    from src.analysis.economic_forecasting import EconomicForecaster
    FORECASTING_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Economic forecasting module not available: {e}")
    FORECASTING_AVAILABLE = False

try:
    from src.analysis.economic_segmentation import EconomicSegmentation
    SEGMENTATION_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Economic segmentation module not available: {e}")
    SEGMENTATION_AVAILABLE = False

try:
    from src.analysis.statistical_modeling import StatisticalModeling
    STATISTICAL_MODELING_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Statistical modeling module not available: {e}")
    STATISTICAL_MODELING_AVAILABLE = False

try:
    from src.core.enhanced_fred_client import EnhancedFREDClient
    ENHANCED_FRED_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Enhanced FRED client not available: {e}")
    ENHANCED_FRED_AVAILABLE = False

try:
    from src.analysis.mathematical_fixes import MathematicalFixes
    MATHEMATICAL_FIXES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Mathematical fixes module not available: {e}")
    MATHEMATICAL_FIXES_AVAILABLE = False

try:
    from src.analysis.alignment_divergence_analyzer import AlignmentDivergenceAnalyzer
    ALIGNMENT_ANALYZER_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Alignment divergence analyzer not available: {e}")
    ALIGNMENT_ANALYZER_AVAILABLE = False

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
        if not ENHANCED_FRED_AVAILABLE:
            raise ImportError("Enhanced FRED client is required but not available")
        
        self.client = EnhancedFREDClient(api_key)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize analytics modules
        self.forecaster = None
        self.segmentation = None
        self.statistical_modeling = None
        
        if MATHEMATICAL_FIXES_AVAILABLE:
            self.mathematical_fixes = MathematicalFixes()
        else:
            self.mathematical_fixes = None
            logger.warning("Mathematical fixes not available - some features may be limited")
        
        # Results storage
        self.data = None
        self.raw_data = None
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
            Dictionary containing all analysis results
        """
        try:
            # Step 1: Data Collection
            self.raw_data = self.client.fetch_economic_data(
                indicators=indicators,
                start_date=start_date,
                end_date=end_date,
                frequency='auto'
            )
            
            # Step 2: Apply Mathematical Fixes
            if self.mathematical_fixes is not None:
                self.data, fix_info = self.mathematical_fixes.apply_comprehensive_fixes(
                    self.raw_data,
                    target_freq='Q',
                    growth_method='pct_change',
                    normalize_units=True,
                    preserve_absolute_values=True  # Preserve absolute values for display
                )
                self.results['mathematical_fixes'] = fix_info
            else:
                logger.warning("Skipping mathematical fixes - module not available")
                self.data = self.raw_data

            # Step 2.5: Alignment & Divergence Analysis (Spearman, Z-score)
            if ALIGNMENT_ANALYZER_AVAILABLE:
                self.alignment_analyzer = AlignmentDivergenceAnalyzer(self.data)
                alignment_results = self.alignment_analyzer.analyze_long_term_alignment()
                zscore_results = self.alignment_analyzer.detect_sudden_deviations()
                self.results['alignment_divergence'] = {
                    'alignment': alignment_results,
                    'zscore_anomalies': zscore_results
                }
            else:
                logger.warning("Skipping alignment analysis - module not available")
                self.results['alignment_divergence'] = {'error': 'Module not available'}
            
            # Step 3: Data Quality Assessment
            quality_report = self.client.validate_data_quality(self.data)
            self.results['data_quality'] = quality_report
            
            # Step 4: Initialize Analytics Modules
            
            if STATISTICAL_MODELING_AVAILABLE:
                self.statistical_modeling = StatisticalModeling(self.data)
            else:
                self.statistical_modeling = None
                logger.warning("Statistical modeling not available")
            
            if FORECASTING_AVAILABLE:
                self.forecaster = EconomicForecaster(self.data)
            else:
                self.forecaster = None
                logger.warning("Economic forecasting not available")
            
            if SEGMENTATION_AVAILABLE:
                self.segmentation = EconomicSegmentation(self.data)
            else:
                self.segmentation = None
                logger.warning("Economic segmentation not available")
            
            # Step 5: Statistical Modeling
            if self.statistical_modeling is not None:
                statistical_results = self._run_statistical_analysis()
                self.results['statistical_modeling'] = statistical_results
            else:
                logger.warning("Skipping statistical modeling - module not available")
                self.results['statistical_modeling'] = {'error': 'Module not available'}
            
            # Step 6: Economic Forecasting
            if self.forecaster is not None:
                forecasting_results = self._run_forecasting_analysis(forecast_periods)
                self.results['forecasting'] = forecasting_results
            else:
                logger.warning("Skipping economic forecasting - module not available")
                self.results['forecasting'] = {'error': 'Module not available'}
            
            # Step 7: Economic Segmentation
            if self.segmentation is not None:
                segmentation_results = self._run_segmentation_analysis()
                self.results['segmentation'] = segmentation_results
            else:
                logger.warning("Skipping economic segmentation - module not available")
                self.results['segmentation'] = {'error': 'Module not available'}
            
            # Step 8: Insights Extraction
            insights = self._extract_insights()
            self.results['insights'] = insights
            
            # Step 9: Generate Reports and Visualizations
            if include_visualizations:
                self._generate_visualizations()
            
            self._generate_comprehensive_report()
            

            return self.results
            
        except Exception as e:
            logger.error(f"Comprehensive analytics pipeline failed: {e}")
            return {'error': f'Comprehensive analytics failed: {str(e)}'}
    
    def _run_statistical_analysis(self) -> Dict:
        """Run statistical modeling analysis"""
        
        if self.statistical_modeling is None:
            return {'error': 'Statistical modeling module not available'}
        
        try:
            # Get available indicators for analysis
            available_indicators = self.data.select_dtypes(include=[np.number]).columns.tolist()
            
            # Ensure we have enough data for analysis
            if len(available_indicators) < 2:
                logger.warning("Insufficient data for statistical analysis")
                return {'error': 'Insufficient data for statistical analysis'}
            
            # Select key indicators for regression analysis
            key_indicators = ['GDPC1', 'INDPRO', 'CPIAUCSL', 'FEDFUNDS', 'UNRATE']
            regression_targets = [ind for ind in key_indicators if ind in available_indicators]
            
            # If we don't have the key indicators, use the first few available
            if not regression_targets and len(available_indicators) >= 2:
                regression_targets = available_indicators[:2]
            
            # Run regression analysis for each target
            regression_results = {}
            for target in regression_targets:
                try:
                    # Get predictors (all other numeric columns)
                    predictors = [ind for ind in available_indicators if ind != target]
                    
                    if len(predictors) > 0:
                        result = self.statistical_modeling.fit_regression_model(target, predictors)
                        regression_results[target] = result
                    else:
                        logger.warning(f"No predictors available for {target}")
                        regression_results[target] = {'error': 'No predictors available'}
                except Exception as e:
                    logger.warning(f"Regression analysis failed for {target}: {e}")
                    regression_results[target] = {'error': str(e)}
            
            # Run correlation analysis
            try:
                correlation_results = self.statistical_modeling.analyze_correlations(available_indicators)
            except Exception as e:
                logger.warning(f"Correlation analysis failed: {e}")
                correlation_results = {'error': str(e)}
            
            # Run Granger causality tests
            causality_results = {}
            if len(regression_targets) >= 2:
                try:
                    # Test causality between first two indicators
                    target1, target2 = regression_targets[:2]
                    causality_result = self.statistical_modeling.perform_granger_causality(target1, target2)
                    causality_results[f"{target1}_vs_{target2}"] = causality_result
                except Exception as e:
                    logger.warning(f"Granger causality test failed: {e}")
                    causality_results['error'] = str(e)
            
            return {
                'correlation': correlation_results,
                'regression': regression_results,
                'causality': causality_results
            }
            
        except Exception as e:
            logger.error(f"Statistical analysis failed: {e}")
            return {'error': str(e)}
    
    def _run_forecasting_analysis(self, forecast_periods: int) -> Dict:
        """Run economic forecasting analysis"""
        
        if self.forecaster is None:
            return {'error': 'Economic forecasting module not available'}
        
        try:
            # Get available indicators for forecasting
            available_indicators = self.data.select_dtypes(include=[np.number]).columns.tolist()
            
            # Select key indicators for forecasting
            key_indicators = ['GDPC1', 'INDPRO', 'RSAFS', 'CPIAUCSL', 'FEDFUNDS', 'DGS10']
            forecast_targets = [ind for ind in key_indicators if ind in available_indicators]
            
            # If we don't have the key indicators, use available ones
            if not forecast_targets and len(available_indicators) > 0:
                forecast_targets = available_indicators[:3]  # Use first 3 available
            
            forecasting_results = {}
            
            for target in forecast_targets:
                try:
                    # Get the time series data for this indicator
                    series_data = self.data[target].dropna()
                    
                    if len(series_data) >= 12:  # Need at least 12 observations
                        result = self.forecaster.forecast_series(
                            series=series_data,
                            model_type='auto',
                            forecast_periods=forecast_periods
                        )
                        # Patch: Robustly handle confidence intervals
                        forecast = result.get('forecast')
                        ci = result.get('confidence_intervals')
                        if ci is not None:
                            try:
                                # Try to access the first row to ensure it's a DataFrame
                                if hasattr(ci, 'iloc'):
                                    _ = ci.iloc[0]
                                elif isinstance(ci, (list, np.ndarray)):
                                    _ = ci[0]
                            except Exception as ci_e:
                                logger.warning(f"[PATCH] Confidence interval access error for {target}: {ci_e}")
                        
                        forecasting_results[target] = result
                    else:
                        logger.warning(f"Insufficient data for forecasting {target}: {len(series_data)} observations")
                        forecasting_results[target] = {'error': f'Insufficient data: {len(series_data)} observations'}
                except Exception as e:
                    logger.error(f"[PATCH] Forecasting analysis failed for {target}: {e}")
                    forecasting_results[target] = {'error': str(e)}
            
            return forecasting_results
            
        except Exception as e:
            logger.error(f"Forecasting analysis failed: {e}")
            return {'error': str(e)}
    
    def _run_segmentation_analysis(self) -> Dict:
        """Run segmentation analysis"""
        logger.info("Running segmentation analysis")
        
        if self.segmentation is None:
            return {'error': 'Economic segmentation module not available'}
        
        try:
            # Get available indicators for segmentation
            available_indicators = self.data.select_dtypes(include=[np.number]).columns.tolist()
            
            # Ensure we have enough data for segmentation
            if len(available_indicators) < 2:
                logger.warning("Insufficient data for segmentation analysis")
                return {'error': 'Insufficient data for segmentation analysis'}
            
            # Run time period clustering
            time_period_clusters = {}
            try:
                # Adjust cluster count based on available data
                n_clusters = min(3, len(available_indicators))
                time_period_clusters = self.segmentation.cluster_time_periods(n_clusters=n_clusters)
            except Exception as e:
                logger.warning(f"Time period clustering failed: {e}")
                time_period_clusters = {'error': str(e)}
            
            # Run series clustering
            series_clusters = {}
            try:
                # Check if we have enough samples for clustering
                available_indicators = self.data.select_dtypes(include=[np.number]).columns.tolist()
                if len(available_indicators) >= 4:
                    series_clusters = self.segmentation.cluster_economic_series(n_clusters=4)
                elif len(available_indicators) >= 2:
                    # Use fewer clusters if we have fewer samples
                    n_clusters = min(3, len(available_indicators))
                    series_clusters = self.segmentation.cluster_economic_series(n_clusters=n_clusters)
                else:
                    series_clusters = {'error': 'Insufficient data for series clustering'}
            except Exception as e:
                logger.warning(f"Series clustering failed: {e}")
                series_clusters = {'error': str(e)}
            
            return {
                'time_period_clusters': time_period_clusters,
                'series_clusters': series_clusters
            }
            
        except Exception as e:
            logger.error(f"Segmentation analysis failed: {e}")
            return {'error': str(e)}
    
    def _extract_insights(self) -> Dict:
        """Extract key insights from all analyses"""
        insights = {
            'key_findings': [],
            'economic_indicators': {},
            'forecasting_insights': [],
            'segmentation_insights': [],
            'statistical_insights': []
        }
        
        try:
            # Extract insights from forecasting
            if 'forecasting' in self.results:
                forecasting_results = self.results['forecasting']
                if isinstance(forecasting_results, dict):
                    for indicator, result in forecasting_results.items():
                        if isinstance(result, dict) and 'error' not in result:
                            # Model performance insights
                            backtest = result.get('backtest', {})
                            if isinstance(backtest, dict) and 'error' not in backtest:
                                mape = backtest.get('mape', 0)
                                if mape < 5:
                                    insights['forecasting_insights'].append(
                                        f"{indicator} forecasting completed"
                                    )
                            
                            # Stationarity insights
                            stationarity = result.get('stationarity', {})
                            if isinstance(stationarity, dict) and 'is_stationary' in stationarity:
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
                if isinstance(segmentation_results, dict):
                    # Time period clustering insights
                    if 'time_period_clusters' in segmentation_results:
                        time_clusters = segmentation_results['time_period_clusters']
                        if isinstance(time_clusters, dict) and 'error' not in time_clusters:
                            n_clusters = time_clusters.get('n_clusters', 0)
                            insights['segmentation_insights'].append(
                                f"Time periods clustered into {n_clusters} distinct economic regimes"
                            )
                    
                    # Series clustering insights
                    if 'series_clusters' in segmentation_results:
                        series_clusters = segmentation_results['series_clusters']
                        if isinstance(series_clusters, dict) and 'error' not in series_clusters:
                            n_clusters = series_clusters.get('n_clusters', 0)
                            insights['segmentation_insights'].append(
                                f"Economic series clustered into {n_clusters} groups based on behavior patterns"
                            )
            
            # Extract insights from statistical modeling
            if 'statistical_modeling' in self.results:
                stat_results = self.results['statistical_modeling']
                if isinstance(stat_results, dict):
                    # Correlation insights
                    if 'correlation' in stat_results:
                        corr_results = stat_results['correlation']
                        if isinstance(corr_results, dict):
                            significant_correlations = corr_results.get('significant_correlations', [])
                            
                            if isinstance(significant_correlations, list) and significant_correlations:
                                try:
                                    strongest_corr = significant_correlations[0]
                                    if isinstance(strongest_corr, dict):
                                        insights['statistical_insights'].append(
                                            f"Strongest correlation: {strongest_corr.get('variable1', 'Unknown')} ↔ {strongest_corr.get('variable2', 'Unknown')} "
                                            f"(r={strongest_corr.get('correlation', 0):.3f})"
                                        )
                                except Exception as e:
                                    logger.warning(f"Error processing correlation insights: {e}")
                                    insights['statistical_insights'].append("Correlation analysis completed")
                    
                    # Regression insights
                    if 'regression' in stat_results:
                        reg_results = stat_results['regression']
                        if isinstance(reg_results, dict):
                            for target, result in reg_results.items():
                                if isinstance(result, dict) and 'error' not in result:
                                    try:
                                        # Handle different possible structures for R²
                                        r2 = 0
                                        if 'performance' in result and isinstance(result['performance'], dict):
                                            performance = result['performance']
                                            r2 = performance.get('r2', 0)
                                        elif 'r2' in result:
                                            r2 = result['r2']
                                        elif 'model_performance' in result and isinstance(result['model_performance'], dict):
                                            model_perf = result['model_performance']
                                            r2 = model_perf.get('r2', 0)
                                        
                                        if r2 > 0.7:
                                            insights['statistical_insights'].append(
                                                f"{target} regression model shows strong explanatory power (R² = {r2:.3f})"
                                            )
                                        elif r2 > 0.5:
                                            insights['statistical_insights'].append(
                                                f"{target} regression model shows moderate explanatory power (R² = {r2:.3f})"
                                            )
                                        else:
                                            insights['statistical_insights'].append(
                                                f"{target} regression analysis completed"
                                            )
                                    except Exception as e:
                                        logger.warning(f"Error processing regression insights for {target}: {e}")
                                        insights['statistical_insights'].append(
                                            f"{target} regression analysis completed"
                                        )
            
            # Generate key findings
            insights['key_findings'] = [
                f"Analysis covers {len(self.data.columns)} economic indicators from {self.data.index.min().strftime('%Y-%m')} to {self.data.index.max().strftime('%Y-%m')}",
                f"Dataset contains {len(self.data)} observations with {self.data.shape[0] * self.data.shape[1]} total data points",
                f"Generated {len(insights['forecasting_insights'])} forecasting insights",
                f"Generated {len(insights['segmentation_insights'])} segmentation insights",
                f"Generated {len(insights['statistical_insights'])} statistical insights"
            ]
            
        except Exception as e:
            logger.error(f"Error extracting insights: {e}")
            insights['key_findings'] = ["Analysis completed with some errors in insight extraction"]
        
        return insights
    
    def _generate_visualizations(self):
        """Generate comprehensive visualizations"""
        logger.info("Generating visualizations")
        
        try:
            # Set style
            plt.style.use('default')  # Use default style instead of seaborn-v0_8
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
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
    
    def _plot_time_series(self):
        """Plot time series of economic indicators"""
        try:
            fig, axes = plt.subplots(3, 2, figsize=(15, 12))
            axes = axes.flatten()
            
            key_indicators = ['GDPC1', 'INDPRO', 'RSAFS', 'CPIAUCSL', 'FEDFUNDS', 'DGS10']
            
            for i, indicator in enumerate(key_indicators):
                if indicator in self.data.columns and i < len(axes):
                    series = self.data[indicator].dropna()
                    if not series.empty:
                        axes[i].plot(series.index, series.values, linewidth=1.5)
                        axes[i].set_title(f'{indicator} - {self.client.ECONOMIC_INDICATORS.get(indicator, indicator)}')
                        axes[i].set_xlabel('Date')
                        axes[i].set_ylabel('Value')
                        axes[i].grid(True, alpha=0.3)
                    else:
                        axes[i].text(0.5, 0.5, f'No data for {indicator}', 
                                   ha='center', va='center', transform=axes[i].transAxes)
                else:
                    axes[i].text(0.5, 0.5, f'{indicator} not available', 
                               ha='center', va='center', transform=axes[i].transAxes)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'economic_indicators_time_series.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating time series chart: {e}")
    
    def _plot_correlation_heatmap(self):
        """Plot correlation heatmap"""
        try:
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
                    
        except Exception as e:
            logger.error(f"Error creating correlation heatmap: {e}")
    
    def _plot_forecasting_results(self):
        """Plot forecasting results"""
        try:
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
                                try:
                                    forecast_data = forecast['forecast']
                                    if hasattr(forecast_data, 'index'):
                                        forecast_values = forecast_data
                                    elif isinstance(forecast_data, (list, np.ndarray)):
                                        forecast_values = forecast_data
                                    else:
                                        forecast_values = None
                                    
                                    if forecast_values is not None:
                                        forecast_index = pd.date_range(
                                            start=series.index[-1] + pd.DateOffset(months=3),
                                            periods=len(forecast_values),
                                            freq='Q'
                                        )
                                        axes[i].plot(forecast_index, forecast_values, 'r--', 
                                                   label='Forecast', linewidth=2)
                                except Exception as e:
                                    logger.warning(f"Error plotting forecast for {indicator}: {e}")
                                
                                axes[i].set_title(f'{indicator} - Forecast')
                                axes[i].set_xlabel('Date')
                                axes[i].set_ylabel('Growth Rate')
                                axes[i].legend()
                                axes[i].grid(True, alpha=0.3)
                                i += 1
                    
                    plt.tight_layout()
                    plt.savefig(self.output_dir / 'forecasting_results.png', dpi=300, bbox_inches='tight')
                    plt.close()
                    
        except Exception as e:
            logger.error(f"Error creating forecast chart: {e}")
    
    def _plot_segmentation_results(self):
        """Plot segmentation results"""
        try:
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
                        
        except Exception as e:
            logger.error(f"Error creating clustering chart: {e}")
    
    def _plot_statistical_diagnostics(self):
        """Plot statistical diagnostics"""
        try:
            if 'statistical_modeling' in self.results:
                stat_results = self.results['statistical_modeling']
                
                # Plot regression diagnostics
                if 'regression' in stat_results:
                    reg_results = stat_results['regression']
                    
                    # Create a summary plot of R² values
                    r2_values = {}
                    for target, result in reg_results.items():
                        if isinstance(result, dict) and 'error' not in result:
                            try:
                                r2 = 0
                                if 'performance' in result and isinstance(result['performance'], dict):
                                    r2 = result['performance'].get('r2', 0)
                                elif 'r2' in result:
                                    r2 = result['r2']
                                elif 'model_performance' in result and isinstance(result['model_performance'], dict):
                                    r2 = result['model_performance'].get('r2', 0)
                                
                                r2_values[target] = r2
                            except Exception as e:
                                logger.warning(f"Error extracting R² for {target}: {e}")
                    
                    if r2_values:
                        plt.figure(figsize=(10, 6))
                        targets = list(r2_values.keys())
                        r2_scores = list(r2_values.values())
                        
                        bars = plt.bar(targets, r2_scores, color='skyblue', alpha=0.7)
                        plt.title('Regression Model Performance (R²)')
                        plt.xlabel('Economic Indicators')
                        plt.ylabel('R² Score')
                        plt.ylim(0, 1)
                        
                        # Add value labels on bars
                        for bar, score in zip(bars, r2_scores):
                            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                   f'{score:.3f}', ha='center', va='bottom')
                        
                        plt.tight_layout()
                        plt.savefig(self.output_dir / 'regression_performance.png', dpi=300, bbox_inches='tight')
                        plt.close()
                        
        except Exception as e:
            logger.error(f"Error creating distribution charts: {e}")
    
    def _generate_comprehensive_report(self):
        """Generate comprehensive analysis report"""
        try:
            report_path = self.output_dir / 'comprehensive_analysis_report.txt'
            
            with open(report_path, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("FRED ML - COMPREHENSIVE ECONOMIC ANALYSIS REPORT\n")
                f.write("=" * 80 + "\n\n")
                
                f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Analysis Period: {self.data.index.min().strftime('%Y-%m-%d')} to {self.data.index.max().strftime('%Y-%m-%d')}\n")
                f.write(f"Economic Indicators: {', '.join(self.data.columns)}\n")
                f.write(f"Total Observations: {len(self.data)}\n\n")
                
                # Data Quality Summary
                if 'data_quality' in self.results:
                    f.write("DATA QUALITY SUMMARY:\n")
                    f.write("-" * 40 + "\n")
                    quality = self.results['data_quality']
                    for indicator, metrics in quality.items():
                        if isinstance(metrics, dict):
                            f.write(f"{indicator}:\n")
                            for metric, value in metrics.items():
                                f.write(f"  {metric}: {value}\n")
                    f.write("\n")
                
                # Statistical Modeling Summary
                if 'statistical_modeling' in self.results:
                    f.write("STATISTICAL MODELING SUMMARY:\n")
                    f.write("-" * 40 + "\n")
                    stat_results = self.results['statistical_modeling']
                    
                    if 'regression' in stat_results:
                        f.write("Regression Analysis:\n")
                        for target, result in stat_results['regression'].items():
                            if isinstance(result, dict) and 'error' not in result:
                                f.write(f"  {target}: ")
                                if 'performance' in result:
                                    perf = result['performance']
                                    f.write(f"R² = {perf.get('r2', 0):.3f}\n")
                                else:
                                    f.write("Analysis completed\n")
                    f.write("\n")
                
                # Forecasting Summary
                if 'forecasting' in self.results:
                    f.write("FORECASTING SUMMARY:\n")
                    f.write("-" * 40 + "\n")
                    for indicator, result in self.results['forecasting'].items():
                        if isinstance(result, dict) and 'error' not in result:
                            f.write(f"{indicator}: ")
                            if 'backtest' in result:
                                backtest = result['backtest']
                                mape = backtest.get('mape', 0)
                                f.write(f"MAPE = {mape:.2f}%\n")
                            else:
                                f.write("Forecast generated\n")
                    f.write("\n")
                
                # Insights Summary
                if 'insights' in self.results:
                    f.write("KEY INSIGHTS:\n")
                    f.write("-" * 40 + "\n")
                    insights = self.results['insights']
                    
                    if 'key_findings' in insights:
                        for finding in insights['key_findings']:
                            f.write(f"• {finding}\n")
                    f.write("\n")
                
                f.write("=" * 80 + "\n")
                f.write("END OF REPORT\n")
                f.write("=" * 80 + "\n")
            
            logger.info(f"Comprehensive report generated: {report_path}")
            
        except Exception as e:
            logger.error(f"Error generating comprehensive report: {e}")
    
    def _generate_comprehensive_summary(self) -> str:
        """Generate a comprehensive summary of all analyses"""
        try:
            summary = []
            summary.append("FRED ML - COMPREHENSIVE ANALYSIS SUMMARY")
            summary.append("=" * 60)
            summary.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            summary.append(f"Data Period: {self.data.index.min().strftime('%Y-%m')} to {self.data.index.max().strftime('%Y-%m')}")
            summary.append(f"Indicators Analyzed: {len(self.data.columns)}")
            summary.append(f"Observations: {len(self.data)}")
            summary.append("")
            
            # Add key insights
            if 'insights' in self.results:
                insights = self.results['insights']
                if 'key_findings' in insights:
                    summary.append("KEY FINDINGS:")
                    for finding in insights['key_findings'][:5]:  # Limit to top 5
                        summary.append(f"• {finding}")
                    summary.append("")
            
            return "\n".join(summary)
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return "Analysis completed with some errors" 