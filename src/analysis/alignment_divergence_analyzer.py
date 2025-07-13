#!/usr/bin/env python3
"""
Alignment and Divergence Analyzer
Analyzes long-term alignment/divergence between economic indicators using Spearman correlation
and detects sudden deviations using Z-score analysis.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class AlignmentDivergenceAnalyzer:
    """
    Analyzes long-term alignment/divergence patterns and sudden deviations in economic indicators
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize analyzer with economic data
        
        Args:
            data: DataFrame with economic indicators (time series)
        """
        self.data = data.copy()
        self.results = {}
        
    def analyze_long_term_alignment(self, 
                                  indicators: List[str] = None,
                                  window_sizes: List[int] = [12, 24, 48],
                                  min_periods: int = 8) -> Dict:
        """
        Analyze long-term alignment/divergence using rolling Spearman correlation
        
        Args:
            indicators: List of indicators to analyze. If None, use all numeric columns
            window_sizes: List of rolling window sizes (in periods)
            min_periods: Minimum periods required for correlation calculation
            
        Returns:
            Dictionary with alignment analysis results
        """
        if indicators is None:
            indicators = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        logger.info(f"Analyzing long-term alignment for {len(indicators)} indicators")
        
        # Calculate growth rates for all indicators
        growth_data = self.data[indicators].pct_change().dropna()
        
        # Initialize results
        alignment_results = {
            'rolling_correlations': {},
            'alignment_summary': {},
            'divergence_periods': {},
            'trend_analysis': {}
        }
        
        # Analyze each pair of indicators
        for i, indicator1 in enumerate(indicators):
            for j, indicator2 in enumerate(indicators):
                if i >= j:  # Skip diagonal and avoid duplicates
                    continue
                    
                pair_name = f"{indicator1}_vs_{indicator2}"
                logger.info(f"Analyzing alignment: {pair_name}")
                
                # Get growth rates for this pair
                pair_data = growth_data[[indicator1, indicator2]].dropna()
                
                if len(pair_data) < min_periods:
                    logger.warning(f"Insufficient data for {pair_name}")
                    continue
                
                # Calculate rolling Spearman correlations for different window sizes
                rolling_corrs = {}
                alignment_trends = {}
                
                for window in window_sizes:
                    if window <= len(pair_data):
                        # Calculate rolling Spearman correlation
                        # Note: pandas rolling.corr() doesn't support method parameter
                        # We'll calculate Spearman correlation manually for each window
                        corr_values = []
                        for start_idx in range(len(pair_data) - window + 1):
                            window_data = pair_data.iloc[start_idx:start_idx + window]
                            if len(window_data.dropna()) >= min_periods:
                                corr_val = window_data.corr(method='spearman').iloc[0, 1]
                                if not pd.isna(corr_val):
                                    corr_values.append(corr_val)
                        
                        if corr_values:
                            rolling_corrs[f"window_{window}"] = corr_values
                            
                            # Analyze alignment trend
                            alignment_trends[f"window_{window}"] = self._analyze_correlation_trend(
                                corr_values, pair_name, window
                            )
                
                # Store results
                alignment_results['rolling_correlations'][pair_name] = rolling_corrs
                alignment_results['trend_analysis'][pair_name] = alignment_trends
                
                # Identify divergence periods
                alignment_results['divergence_periods'][pair_name] = self._identify_divergence_periods(
                    pair_data, rolling_corrs, pair_name
                )
        
        # Generate alignment summary
        alignment_results['alignment_summary'] = self._generate_alignment_summary(
            alignment_results['trend_analysis']
        )
        
        self.results['alignment'] = alignment_results
        return alignment_results
    
    def detect_sudden_deviations(self, 
                               indicators: List[str] = None,
                               z_threshold: float = 2.0,
                               window_size: int = 12,
                               min_periods: int = 6) -> Dict:
        """
        Detect sudden deviations using Z-score analysis
        
        Args:
            indicators: List of indicators to analyze. If None, use all numeric columns
            z_threshold: Z-score threshold for flagging deviations
            window_size: Rolling window size for Z-score calculation
            min_periods: Minimum periods required for Z-score calculation
            
        Returns:
            Dictionary with deviation detection results
        """
        if indicators is None:
            indicators = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        logger.info(f"Detecting sudden deviations for {len(indicators)} indicators")
        
        # Calculate growth rates
        growth_data = self.data[indicators].pct_change().dropna()
        
        deviation_results = {
            'z_scores': {},
            'deviations': {},
            'deviation_summary': {},
            'extreme_events': {}
        }
        
        for indicator in indicators:
            if indicator not in growth_data.columns:
                continue
                
            series = growth_data[indicator].dropna()
            
            if len(series) < min_periods:
                logger.warning(f"Insufficient data for {indicator}")
                continue
            
            # Calculate rolling Z-scores
            rolling_mean = series.rolling(window=window_size, min_periods=min_periods).mean()
            rolling_std = series.rolling(window=window_size, min_periods=min_periods).std()
            
            # Calculate Z-scores
            z_scores = (series - rolling_mean) / rolling_std
            
            # Identify deviations
            deviations = z_scores[abs(z_scores) > z_threshold]
            
            # Store results
            deviation_results['z_scores'][indicator] = z_scores
            deviation_results['deviations'][indicator] = deviations
            
            # Analyze extreme events
            deviation_results['extreme_events'][indicator] = self._analyze_extreme_events(
                series, z_scores, deviations, indicator
            )
        
        # Generate deviation summary
        deviation_results['deviation_summary'] = self._generate_deviation_summary(
            deviation_results['deviations'], deviation_results['extreme_events']
        )
        
        self.results['deviations'] = deviation_results
        return deviation_results
    
    def _analyze_correlation_trend(self, corr_values: List[float], 
                                 pair_name: str, window: int) -> Dict:
        """Analyze trend in correlation values"""
        if len(corr_values) < 2:
            return {'trend': 'insufficient_data', 'direction': 'unknown'}
        
        # Calculate trend using linear regression
        x = np.arange(len(corr_values))
        y = np.array(corr_values)
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Determine trend direction and strength
        if abs(slope) < 0.001:
            trend_direction = 'stable'
        elif slope > 0:
            trend_direction = 'increasing_alignment'
        else:
            trend_direction = 'decreasing_alignment'
        
        # Assess trend strength
        if abs(r_value) > 0.7:
            trend_strength = 'strong'
        elif abs(r_value) > 0.4:
            trend_strength = 'moderate'
        else:
            trend_strength = 'weak'
        
        return {
            'trend': trend_direction,
            'strength': trend_strength,
            'slope': slope,
            'r_squared': r_value**2,
            'p_value': p_value,
            'mean_correlation': np.mean(corr_values),
            'correlation_volatility': np.std(corr_values)
        }
    
    def _identify_divergence_periods(self, pair_data: pd.DataFrame, 
                                   rolling_corrs: Dict, pair_name: str) -> Dict:
        """Identify periods of significant divergence"""
        divergence_periods = []
        
        for window_name, corr_values in rolling_corrs.items():
            if len(corr_values) < 4:
                continue
                
            # Find periods where correlation is negative or very low
            corr_series = pd.Series(corr_values)
            divergence_mask = corr_series < 0.1  # Low correlation threshold
            
            if divergence_mask.any():
                divergence_periods.append({
                    'window': window_name,
                    'divergence_count': divergence_mask.sum(),
                    'divergence_percentage': (divergence_mask.sum() / len(corr_series)) * 100,
                    'min_correlation': corr_series.min(),
                    'max_correlation': corr_series.max()
                })
        
        return divergence_periods
    
    def _analyze_extreme_events(self, series: pd.Series, z_scores: pd.Series, 
                              deviations: pd.Series, indicator: str) -> Dict:
        """Analyze extreme events for an indicator"""
        if deviations.empty:
            return {'count': 0, 'events': []}
        
        events = []
        for date, z_score in deviations.items():
            events.append({
                'date': date,
                'z_score': z_score,
                'growth_rate': series.loc[date],
                'severity': 'extreme' if abs(z_score) > 3.0 else 'moderate'
            })
        
        # Sort by absolute Z-score
        events.sort(key=lambda x: abs(x['z_score']), reverse=True)
        
        return {
            'count': len(events),
            'events': events[:10],  # Top 10 most extreme events
            'max_z_score': max(abs(d['z_score']) for d in events),
            'mean_z_score': np.mean([abs(d['z_score']) for d in events])
        }
    
    def _generate_alignment_summary(self, trend_analysis: Dict) -> Dict:
        """Generate summary of alignment trends"""
        summary = {
            'increasing_alignment': [],
            'decreasing_alignment': [],
            'stable_alignment': [],
            'strong_trends': [],
            'moderate_trends': [],
            'weak_trends': []
        }
        
        for pair_name, trends in trend_analysis.items():
            for window_name, trend_info in trends.items():
                trend = trend_info['trend']
                strength = trend_info['strength']
                
                if trend == 'increasing_alignment':
                    summary['increasing_alignment'].append(pair_name)
                elif trend == 'decreasing_alignment':
                    summary['decreasing_alignment'].append(pair_name)
                elif trend == 'stable':
                    summary['stable_alignment'].append(pair_name)
                
                if strength == 'strong':
                    summary['strong_trends'].append(f"{pair_name}_{window_name}")
                elif strength == 'moderate':
                    summary['moderate_trends'].append(f"{pair_name}_{window_name}")
                else:
                    summary['weak_trends'].append(f"{pair_name}_{window_name}")
        
        return summary
    
    def _generate_deviation_summary(self, deviations: Dict, extreme_events: Dict) -> Dict:
        """Generate summary of deviation analysis"""
        summary = {
            'total_deviations': 0,
            'indicators_with_deviations': [],
            'most_volatile_indicators': [],
            'extreme_events_count': 0
        }
        
        for indicator, dev_series in deviations.items():
            if not dev_series.empty:
                summary['total_deviations'] += len(dev_series)
                summary['indicators_with_deviations'].append(indicator)
                
                # Calculate volatility (standard deviation of growth rates)
                growth_series = self.data[indicator].pct_change().dropna()
                volatility = growth_series.std()
                
                summary['most_volatile_indicators'].append({
                    'indicator': indicator,
                    'volatility': volatility,
                    'deviation_count': len(dev_series)
                })
        
        # Sort by volatility
        summary['most_volatile_indicators'].sort(
            key=lambda x: x['volatility'], reverse=True
        )
        
        # Count extreme events
        for indicator, events in extreme_events.items():
            summary['extreme_events_count'] += events['count']
        
        return summary
    
    def plot_alignment_analysis(self, save_path: Optional[str] = None) -> None:
        """Plot alignment analysis results"""
        if 'alignment' not in self.results:
            logger.warning("No alignment analysis results to plot")
            return
        
        alignment_results = self.results['alignment']
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Economic Indicators Alignment Analysis', fontsize=16)
        
        # Plot 1: Rolling correlations heatmap
        if alignment_results['rolling_correlations']:
            # Create correlation matrix for latest values
            latest_correlations = {}
            for pair_name, windows in alignment_results['rolling_correlations'].items():
                if 'window_12' in windows and windows['window_12']:
                    latest_correlations[pair_name] = windows['window_12'][-1]
            
            if latest_correlations:
                # Convert to matrix format
                indicators = list(set([pair.split('_vs_')[0] for pair in latest_correlations.keys()] + 
                                   [pair.split('_vs_')[1] for pair in latest_correlations.keys()]))
                
                corr_matrix = pd.DataFrame(index=indicators, columns=indicators, dtype=float)
                for pair, corr in latest_correlations.items():
                    ind1, ind2 = pair.split('_vs_')
                    corr_matrix.loc[ind1, ind2] = float(corr)
                    corr_matrix.loc[ind2, ind1] = float(corr)
                
                # Fill diagonal with 1
                np.fill_diagonal(corr_matrix.values, 1.0)
                
                # Ensure all values are numeric
                corr_matrix = corr_matrix.astype(float)
                
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                           ax=axes[0,0], cbar_kws={'label': 'Spearman Correlation'})
                axes[0,0].set_title('Latest Rolling Correlations (12-period window)')
        
        # Plot 2: Alignment trends
        if alignment_results['trend_analysis']:
            trend_data = []
            for pair_name, trends in alignment_results['trend_analysis'].items():
                for window_name, trend_info in trends.items():
                    trend_data.append({
                        'Pair': pair_name,
                        'Window': window_name,
                        'Trend': trend_info['trend'],
                        'Strength': trend_info['strength'],
                        'Slope': trend_info['slope']
                    })
            
            if trend_data:
                trend_df = pd.DataFrame(trend_data)
                trend_counts = trend_df['Trend'].value_counts()
                
                axes[0,1].pie(trend_counts.values, labels=trend_counts.index, autopct='%1.1f%%')
                axes[0,1].set_title('Alignment Trend Distribution')
        
        # Plot 3: Deviation summary
        if 'deviations' in self.results:
            deviation_results = self.results['deviations']
            if deviation_results['deviation_summary']['most_volatile_indicators']:
                vol_data = deviation_results['deviation_summary']['most_volatile_indicators']
                indicators = [d['indicator'] for d in vol_data[:5]]
                volatilities = [d['volatility'] for d in vol_data[:5]]
                
                axes[1,0].bar(indicators, volatilities)
                axes[1,0].set_title('Most Volatile Indicators')
                axes[1,0].set_ylabel('Volatility (Std Dev of Growth Rates)')
                axes[1,0].tick_params(axis='x', rotation=45)
        
        # Plot 4: Z-score timeline
        if 'deviations' in self.results:
            deviation_results = self.results['deviations']
            if deviation_results['z_scores']:
                # Plot Z-scores for first few indicators
                indicators_to_plot = list(deviation_results['z_scores'].keys())[:3]
                
                for indicator in indicators_to_plot:
                    z_scores = deviation_results['z_scores'][indicator]
                    axes[1,1].plot(z_scores.index, z_scores.values, label=indicator, alpha=0.7)
                
                axes[1,1].axhline(y=2, color='red', linestyle='--', alpha=0.5, label='Threshold')
                axes[1,1].axhline(y=-2, color='red', linestyle='--', alpha=0.5)
                axes[1,1].set_title('Z-Score Timeline')
                axes[1,1].set_ylabel('Z-Score')
                axes[1,1].legend()
                axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_insights_report(self) -> str:
        """Generate a comprehensive insights report"""
        if not self.results:
            return "No analysis results available. Please run alignment and deviation analysis first."
        
        report = []
        report.append("=" * 80)
        report.append("ECONOMIC INDICATORS ALIGNMENT & DEVIATION ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Alignment insights
        if 'alignment' in self.results:
            alignment_results = self.results['alignment']
            summary = alignment_results['alignment_summary']
            
            report.append("📊 LONG-TERM ALIGNMENT ANALYSIS")
            report.append("-" * 40)
            
            report.append(f"• Increasing Alignment Pairs: {len(summary['increasing_alignment'])}")
            report.append(f"• Decreasing Alignment Pairs: {len(summary['decreasing_alignment'])}")
            report.append(f"• Stable Alignment Pairs: {len(summary['stable_alignment'])}")
            report.append(f"• Strong Trends: {len(summary['strong_trends'])}")
            report.append("")
            
            if summary['increasing_alignment']:
                report.append("🔺 Pairs with Increasing Alignment:")
                for pair in summary['increasing_alignment'][:5]:
                    report.append(f"  - {pair}")
                report.append("")
            
            if summary['decreasing_alignment']:
                report.append("🔻 Pairs with Decreasing Alignment:")
                for pair in summary['decreasing_alignment'][:5]:
                    report.append(f"  - {pair}")
                report.append("")
        
        # Deviation insights
        if 'deviations' in self.results:
            deviation_results = self.results['deviations']
            summary = deviation_results['deviation_summary']
            
            report.append("⚠️ SUDDEN DEVIATION ANALYSIS")
            report.append("-" * 35)
            
            report.append(f"• Total Deviations Detected: {summary['total_deviations']}")
            report.append(f"• Indicators with Deviations: {len(summary['indicators_with_deviations'])}")
            report.append(f"• Extreme Events: {summary['extreme_events_count']}")
            report.append("")
            
            if summary['most_volatile_indicators']:
                report.append("📈 Most Volatile Indicators:")
                for item in summary['most_volatile_indicators'][:5]:
                    report.append(f"  - {item['indicator']}: {item['volatility']:.4f} volatility")
                report.append("")
            
            # Show extreme events
            extreme_events = deviation_results['extreme_events']
            if extreme_events:
                report.append("🚨 Recent Extreme Events:")
                for indicator, events in extreme_events.items():
                    if events['events']:
                        latest_event = events['events'][0]
                        report.append(f"  - {indicator}: {latest_event['date'].strftime('%Y-%m-%d')} "
                                   f"(Z-score: {latest_event['z_score']:.2f})")
                report.append("")
        
        report.append("=" * 80)
        report.append("Analysis completed successfully.")
        
        return "\n".join(report) 