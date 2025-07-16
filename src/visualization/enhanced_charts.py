"""
Enhanced Visualization Module
Shows mathematical fixes and advanced analytics in action
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging

logger = logging.getLogger(__name__)

class EnhancedChartGenerator:
    """
    Enhanced chart generator with mathematical fixes visualization
    """
    
    def __init__(self):
        """Initialize enhanced chart generator"""
        self.colors = {
            'primary': '#1e3c72',
            'secondary': '#2a5298',
            'accent': '#ff6b6b',
            'success': '#51cf66',
            'warning': '#ffd43b',
            'info': '#74c0fc'
        }
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def create_mathematical_fixes_comparison(self, raw_data: pd.DataFrame, 
                                           fixed_data: pd.DataFrame,
                                           fix_info: Dict) -> go.Figure:
        """
        Create comparison chart showing before/after mathematical fixes
        
        Args:
            raw_data: Original data
            fixed_data: Data after mathematical fixes
            fix_info: Information about applied fixes
            
        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Before: Raw Data', 'After: Unit Normalization',
                          'Before: Mixed Frequencies', 'After: Aligned Frequencies'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Sample a few indicators for visualization
        indicators = list(raw_data.columns)[:4]
        
        # Before/After raw data
        for i, indicator in enumerate(indicators):
            if indicator in raw_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=raw_data.index,
                        y=raw_data[indicator],
                        name=f'{indicator} (Raw)',
                        line=dict(color=self.colors['primary']),
                        showlegend=(i == 0)
                    ),
                    row=1, col=1
                )
        
        # Before/After unit normalization
        for i, indicator in enumerate(indicators):
            if indicator in fixed_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=fixed_data.index,
                        y=fixed_data[indicator],
                        name=f'{indicator} (Normalized)',
                        line=dict(color=self.colors['success']),
                        showlegend=(i == 0)
                    ),
                    row=1, col=2
                )
        
        # Before/After frequency alignment
        for i, indicator in enumerate(indicators):
            if indicator in raw_data.columns:
                # Show original frequency
                fig.add_trace(
                    go.Scatter(
                        x=raw_data.index,
                        y=raw_data[indicator],
                        name=f'{indicator} (Original)',
                        line=dict(color=self.colors['warning']),
                        showlegend=(i == 0)
                    ),
                    row=2, col=1
                )
        
        # After frequency alignment
        for i, indicator in enumerate(indicators):
            if indicator in fixed_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=fixed_data.index,
                        y=fixed_data[indicator],
                        name=f'{indicator} (Aligned)',
                        line=dict(color=self.colors['info']),
                        showlegend=(i == 0)
                    ),
                    row=2, col=2
                )
        
        fig.update_layout(
            title="Mathematical Fixes: Before vs After",
            height=600,
            showlegend=True
        )
        
        return fig
    
    def create_growth_rate_analysis(self, data: pd.DataFrame, 
                                  method: str = 'pct_change') -> go.Figure:
        """
        Create growth rate analysis chart
        
        Args:
            data: Economic data
            method: Growth calculation method
            
        Returns:
            Plotly figure
        """
        # Calculate growth rates
        if method == 'pct_change':
            growth_data = data.pct_change() * 100
        else:
            growth_data = np.log(data / data.shift(1)) * 100
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Growth Rates Over Time', 'Growth Rate Distribution',
                          'Cumulative Growth', 'Growth Rate Volatility'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Growth rates over time
        for indicator in data.columns:
            if indicator in growth_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=growth_data.index,
                        y=growth_data[indicator],
                        name=indicator,
                        mode='lines'
                    ),
                    row=1, col=1
                )
        
        # Growth rate distribution
        for indicator in data.columns:
            if indicator in growth_data.columns:
                fig.add_trace(
                    go.Histogram(
                        x=growth_data[indicator].dropna(),
                        name=indicator,
                        opacity=0.7
                    ),
                    row=1, col=2
                )
        
        # Cumulative growth
        cumulative_growth = (1 + growth_data / 100).cumprod()
        for indicator in data.columns:
            if indicator in cumulative_growth.columns:
                fig.add_trace(
                    go.Scatter(
                        x=cumulative_growth.index,
                        y=cumulative_growth[indicator],
                        name=indicator,
                        mode='lines'
                    ),
                    row=2, col=1
                )
        
        # Growth rate volatility (rolling std)
        volatility = growth_data.rolling(window=12).std()
        for indicator in data.columns:
            if indicator in volatility.columns:
                fig.add_trace(
                    go.Scatter(
                        x=volatility.index,
                        y=volatility[indicator],
                        name=indicator,
                        mode='lines'
                    ),
                    row=2, col=2
                )
        
        fig.update_layout(
            title=f"Growth Rate Analysis ({method})",
            height=600,
            showlegend=True
        )
        
        return fig
    
    def create_forecast_accuracy_chart(self, actual: pd.Series, 
                                     forecast: pd.Series,
                                     title: str = "Forecast Accuracy") -> go.Figure:
        """
        Create forecast accuracy chart with error metrics
        
        Args:
            actual: Actual values
            forecast: Forecasted values
            title: Chart title
            
        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Actual vs Forecast', 'Forecast Errors',
                          'Error Distribution', 'Cumulative Error'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Actual vs Forecast
        fig.add_trace(
            go.Scatter(
                x=actual.index,
                y=actual.values,
                name='Actual',
                line=dict(color=self.colors['primary'])
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=forecast.index,
                y=forecast.values,
                name='Forecast',
                line=dict(color=self.colors['accent'])
            ),
            row=1, col=1
        )
        
        # Forecast errors
        errors = actual - forecast
        fig.add_trace(
            go.Scatter(
                x=errors.index,
                y=errors.values,
                name='Errors',
                line=dict(color=self.colors['warning'])
            ),
            row=1, col=2
        )
        
        # Error distribution
        fig.add_trace(
            go.Histogram(
                x=errors.values,
                name='Error Distribution',
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # Cumulative error
        cumulative_error = errors.cumsum()
        fig.add_trace(
            go.Scatter(
                x=cumulative_error.index,
                y=cumulative_error.values,
                name='Cumulative Error',
                line=dict(color=self.colors['info'])
            ),
            row=2, col=2
        )
        
        # Calculate error metrics
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors**2))
        mape = np.mean(np.abs(errors / np.maximum(np.abs(actual), 1e-8))) * 100
        
        fig.update_layout(
            title=f"{title}<br><sub>MAE: {mae:.2f} | RMSE: {rmse:.2f} | MAPE: {mape:.2f}%</sub>",
            height=600,
            showlegend=True
        )
        
        return fig
    
    def create_correlation_heatmap_enhanced(self, data: pd.DataFrame,
                                         method: str = 'pearson') -> go.Figure:
        """
        Create enhanced correlation heatmap
        
        Args:
            data: Economic data
            method: Correlation method
            
        Returns:
            Plotly figure
        """
        # Calculate correlation matrix
        corr_matrix = data.corr(method=method)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 3),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=f"Economic Indicators Correlation Matrix ({method})",
            xaxis_title="Indicators",
            yaxis_title="Indicators",
            height=600
        )
        
        return fig
    
    def create_segmentation_visualization(self, data: pd.DataFrame,
                                       cluster_labels: np.ndarray,
                                       method: str = 'PCA') -> go.Figure:
        """
        Create segmentation visualization
        
        Args:
            data: Economic data
            cluster_labels: Cluster labels
            method: Dimensionality reduction method
            
        Returns:
            Plotly figure
        """
        if method == 'PCA':
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            
            # Standardize data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data.dropna())
            
            # Apply PCA
            pca = PCA(n_components=2)
            pca_data = pca.fit_transform(scaled_data)
            
            # Create scatter plot
            fig = px.scatter(
                x=pca_data[:, 0],
                y=pca_data[:, 1],
                color=cluster_labels,
                title=f"Economic Segmentation ({method})",
                labels={'x': f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)',
                       'y': f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)'}
            )
            
            fig.update_layout(height=500)
            
        else:
            # Fallback to first two dimensions
            fig = px.scatter(
                x=data.iloc[:, 0],
                y=data.iloc[:, 1],
                color=cluster_labels,
                title=f"Economic Segmentation ({method})"
            )
        
        return fig
    
    def create_comprehensive_dashboard(self, raw_data: pd.DataFrame,
                                    fixed_data: pd.DataFrame,
                                    results: Dict) -> go.Figure:
        """
        Create comprehensive dashboard with all visualizations
        
        Args:
            raw_data: Original data
            fixed_data: Data after fixes
            results: Analysis results
            
        Returns:
            Plotly figure
        """
        # Create subplots for comprehensive dashboard
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Raw Data Overview', 'Fixed Data Overview',
                          'Growth Rate Analysis', 'Correlation Matrix',
                          'Forecast Results', 'Segmentation Results'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Raw data overview
        for indicator in raw_data.columns[:3]:  # Show first 3 indicators
            fig.add_trace(
                go.Scatter(
                    x=raw_data.index,
                    y=raw_data[indicator],
                    name=f'{indicator} (Raw)',
                    mode='lines'
                ),
                row=1, col=1
            )
        
        # Fixed data overview
        for indicator in fixed_data.columns[:3]:  # Show first 3 indicators
            fig.add_trace(
                go.Scatter(
                    x=fixed_data.index,
                    y=fixed_data[indicator],
                    name=f'{indicator} (Fixed)',
                    mode='lines'
                ),
                row=1, col=2
            )
        
        # Growth rate analysis
        growth_data = fixed_data.pct_change() * 100
        for indicator in growth_data.columns[:2]:  # Show first 2 indicators
            fig.add_trace(
                go.Scatter(
                    x=growth_data.index,
                    y=growth_data[indicator],
                    name=f'{indicator} Growth',
                    mode='lines'
                ),
                row=2, col=1
            )
        
        # Correlation matrix (simplified)
        corr_matrix = fixed_data.corr()
        fig.add_trace(
            go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale='RdBu',
                zmid=0
            ),
            row=2, col=2
        )
        
        # Forecast results (if available)
        if 'forecasting' in results:
            forecasting_results = results['forecasting']
            for indicator, result in forecasting_results.items():
                if 'error' not in result and 'forecast' in result:
                    forecast_data = result['forecast']
                    if 'forecast' in forecast_data:
                        fig.add_trace(
                            go.Scatter(
                                x=forecast_data.get('forecast_index', []),
                                y=forecast_data['forecast'],
                                name=f'{indicator} Forecast',
                                mode='lines',
                                line=dict(dash='dash')
                            ),
                            row=3, col=1
                        )
        
        # Segmentation results (if available)
        if 'segmentation' in results:
            segmentation_results = results['segmentation']
            if 'time_period_clusters' in segmentation_results:
                time_clusters = segmentation_results['time_period_clusters']
                if 'cluster_labels' in time_clusters:
                    cluster_labels = time_clusters['cluster_labels']
                    fig.add_trace(
                        go.Scatter(
                            x=list(range(len(cluster_labels))),
                            y=cluster_labels,
                            mode='markers',
                            name='Time Clusters',
                            marker=dict(size=8)
                        ),
                        row=3, col=2
                    )
        
        fig.update_layout(
            title="Comprehensive Economic Analytics Dashboard",
            height=900,
            showlegend=True
        )
        
        return fig 

    def create_spearman_alignment_heatmap(self, alignment_results):
        """Create a heatmap of average rolling Spearman correlations for all pairs."""
        # Extract mean correlations for each pair and window
        pair_means = {}
        for pair, windows in alignment_results.get('rolling_correlations', {}).items():
            for window, corrs in windows.items():
                pair_means[(pair, window)] = np.mean(corrs) if corrs else np.nan
        # Convert to DataFrame for heatmap
        if not pair_means:
            return go.Figure()
        df = pd.DataFrame.from_dict(pair_means, orient='index', columns=['mean_corr'])
        df = df.reset_index()
        df[['pair', 'window']] = pd.DataFrame(df['index'].tolist(), index=df.index)
        heatmap_df = df.pivot(index='pair', columns='window', values='mean_corr')
        fig = px.imshow(heatmap_df, text_auto=True, color_continuous_scale='RdBu_r',
                        aspect='auto', title='Average Rolling Spearman Correlation')
        fig.update_layout(height=600)
        return fig

    def create_rolling_spearman_plot(self, alignment_results, pair, window):
        """Plot rolling Spearman correlation for a given pair and window size."""
        corrs = alignment_results.get('rolling_correlations', {}).get(pair, {}).get(window, [])
        if not corrs:
            return go.Figure()
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=corrs, mode='lines', name=f'{pair} ({window})'))
        fig.update_layout(title=f'Rolling Spearman Correlation: {pair} ({window})',
                          xaxis_title='Window Index', yaxis_title='Spearman Correlation', height=400)
        return fig

    def create_zscore_anomaly_chart(self, zscore_results, indicator):
        """Plot Z-score time series and highlight anomalies for a given indicator."""
        z_scores = zscore_results.get('z_scores', {}).get(indicator, None)
        deviations = zscore_results.get('deviations', {}).get(indicator, None)
        if z_scores is None or deviations is None:
            return go.Figure()
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=z_scores, mode='lines', name='Z-score'))
        # Highlight anomalies
        if not deviations.empty:
            fig.add_trace(go.Scatter(x=deviations.index, y=deviations.values, mode='markers',
                                     marker=dict(color='red', size=8), name='Anomaly'))
        fig.update_layout(title=f'Z-score Anomalies: {indicator}',
                          xaxis_title='Time', yaxis_title='Z-score', height=400)
        return fig 