#!/usr/bin/env python3
"""
Local Chart Generator for FRED ML
Creates comprehensive economic visualizations and stores them locally
"""

import io
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Add parent directory to path for config import
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Also add the project root (two levels up from src)
project_root = os.path.dirname(parent_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Use hardcoded defaults to avoid import issues
DEFAULT_OUTPUT_DIR = 'data/processed'
DEFAULT_PLOTS_DIR = 'data/exports'

# Set style for matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class LocalChartGenerator:
    """Generate comprehensive economic visualizations locally"""
    
    def __init__(self, output_dir: str = None):
        if output_dir is None:
            # Use absolute path to avoid relative path issues
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            output_dir = os.path.join(project_root, DEFAULT_PLOTS_DIR, 'visualizations')
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.chart_paths = []
        
    def create_time_series_chart(self, df: pd.DataFrame, title: str = "Economic Indicators") -> str:
        """Create time series chart and save locally"""
        try:
            fig, ax = plt.subplots(figsize=(15, 8))
            
            for column in df.columns:
                if column != 'Date':
                    ax.plot(df.index, df[column], label=column, linewidth=2)
            
            ax.set_title(title, fontsize=16, fontweight='bold')
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Value', fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save locally
            chart_filename = f"time_series_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            chart_path = os.path.join(self.output_dir, chart_filename)
            plt.savefig(chart_path, format='png', dpi=300, bbox_inches='tight')
            
            plt.close()
            self.chart_paths.append(chart_path)
            return chart_path
            
        except Exception as e:
            print(f"Error creating time series chart: {e}")
            return None
    
    def create_correlation_heatmap(self, df: pd.DataFrame) -> str:
        """Create correlation heatmap and save locally"""
        try:
            corr_matrix = df.corr()
            
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                       square=True, linewidths=0.5, cbar_kws={"shrink": .8})
            
            plt.title('Economic Indicators Correlation Matrix', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # Save locally
            chart_filename = f"correlation_heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            chart_path = os.path.join(self.output_dir, chart_filename)
            plt.savefig(chart_path, format='png', dpi=300, bbox_inches='tight')
            
            plt.close()
            self.chart_paths.append(chart_path)
            return chart_path
            
        except Exception as e:
            print(f"Error creating correlation heatmap: {e}")
            return None
    
    def create_distribution_charts(self, df: pd.DataFrame) -> List[str]:
        """Create distribution charts for each indicator"""
        chart_paths = []
        
        try:
            for column in df.columns:
                if column != 'Date':
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Histogram with KDE
                    sns.histplot(df[column].dropna(), kde=True, ax=ax)
                    ax.set_title(f'Distribution of {column}', fontsize=14, fontweight='bold')
                    ax.set_xlabel(column, fontsize=12)
                    ax.set_ylabel('Frequency', fontsize=12)
                    plt.tight_layout()
                    
                    # Save locally
                    chart_filename = f"distribution_{column}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    chart_path = os.path.join(self.output_dir, chart_filename)
                    plt.savefig(chart_path, format='png', dpi=300, bbox_inches='tight')
                    
                    plt.close()
                    chart_paths.append(chart_path)
                    self.chart_paths.append(chart_path)
            
            return chart_paths
            
        except Exception as e:
            print(f"Error creating distribution charts: {e}")
            return []
    
    def create_pca_visualization(self, df: pd.DataFrame, n_components: int = 2) -> str:
        """Create PCA visualization and save locally"""
        try:
            # Prepare data
            df_clean = df.dropna()
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(df_clean)
            
            # Perform PCA
            pca = PCA(n_components=n_components)
            pca_result = pca.fit_transform(scaled_data)
            
            # Create visualization
            fig, ax = plt.subplots(figsize=(12, 8))
            
            if n_components == 2:
                scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.6)
                ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
                ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
            else:
                # For 3D or more, show first two components
                scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.6)
                ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
                ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
            
            ax.set_title('PCA Visualization of Economic Indicators', fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save locally
            chart_filename = f"pca_visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            chart_path = os.path.join(self.output_dir, chart_filename)
            plt.savefig(chart_path, format='png', dpi=300, bbox_inches='tight')
            
            plt.close()
            self.chart_paths.append(chart_path)
            return chart_path
            
        except Exception as e:
            print(f"Error creating PCA visualization: {e}")
            return None
    
    def create_forecast_chart(self, historical_data: pd.Series, forecast_data: List[float], 
                            title: str = "Economic Forecast") -> str:
        """Create forecast chart and save locally"""
        try:
            fig, ax = plt.subplots(figsize=(15, 8))
            
            # Plot historical data
            ax.plot(historical_data.index, historical_data.values, 
                   label='Historical', linewidth=2, color='blue')
            
            # Plot forecast
            forecast_index = pd.date_range(
                start=historical_data.index[-1] + pd.DateOffset(months=1),
                periods=len(forecast_data),
                freq='M'
            )
            ax.plot(forecast_index, forecast_data, 
                   label='Forecast', linewidth=2, color='red', linestyle='--')
            
            ax.set_title(title, fontsize=16, fontweight='bold')
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Value', fontsize=12)
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save locally
            chart_filename = f"forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            chart_path = os.path.join(self.output_dir, chart_filename)
            plt.savefig(chart_path, format='png', dpi=300, bbox_inches='tight')
            
            plt.close()
            self.chart_paths.append(chart_path)
            return chart_path
            
        except Exception as e:
            print(f"Error creating forecast chart: {e}")
            return None
    
    def create_clustering_chart(self, df: pd.DataFrame, n_clusters: int = 3) -> str:
        """Create clustering visualization and save locally"""
        try:
            from sklearn.cluster import KMeans
            
            # Prepare data
            df_clean = df.dropna()
            # Check for sufficient data
            if df_clean.empty or df_clean.shape[0] < n_clusters or df_clean.shape[1] < 2:
                print(f"Error creating clustering chart: Not enough data for clustering (rows: {df_clean.shape[0]}, cols: {df_clean.shape[1]})")
                return None
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(df_clean)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(scaled_data)
            
            # PCA for visualization
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(scaled_data)
            
            # Create visualization
            fig, ax = plt.subplots(figsize=(12, 8))
            
            scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], 
                               c=clusters, cmap='viridis', alpha=0.6)
            
            # Add cluster centers
            centers_pca = pca.transform(kmeans.cluster_centers_)
            ax.scatter(centers_pca[:, 0], centers_pca[:, 1], 
                      c='red', marker='x', s=200, linewidths=3, label='Cluster Centers')
            
            ax.set_title(f'K-Means Clustering (k={n_clusters})', fontsize=16, fontweight='bold')
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save locally
            chart_filename = f"clustering_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            chart_path = os.path.join(self.output_dir, chart_filename)
            plt.savefig(chart_path, format='png', dpi=300, bbox_inches='tight')
            
            plt.close()
            self.chart_paths.append(chart_path)
            return chart_path
            
        except Exception as e:
            print(f"Error creating clustering chart: {e}")
            return None
    
    def generate_comprehensive_visualizations(self, df: pd.DataFrame, analysis_type: str = "comprehensive") -> Dict[str, str]:
        """Generate comprehensive visualizations based on analysis type"""
        visualizations = {}
        
        try:
            # Always create time series and correlation charts
            visualizations['time_series'] = self.create_time_series_chart(df)
            visualizations['correlation'] = self.create_correlation_heatmap(df)
            visualizations['distributions'] = self.create_distribution_charts(df)
            
            if analysis_type in ["comprehensive", "statistical"]:
                # Add PCA visualization
                visualizations['pca'] = self.create_pca_visualization(df)
                
                # Add clustering
                visualizations['clustering'] = self.create_clustering_chart(df)
            
            if analysis_type in ["comprehensive", "forecasting"]:
                # Add forecast visualization (using sample data)
                sample_series = df.iloc[:, 0] if not df.empty else pd.Series([1, 2, 3, 4, 5])
                sample_forecast = [sample_series.iloc[-1] * 1.02, sample_series.iloc[-1] * 1.04]
                visualizations['forecast'] = self.create_forecast_chart(sample_series, sample_forecast)
            
            # Store visualization metadata
            metadata = {
                'analysis_type': analysis_type,
                'timestamp': datetime.now().isoformat(),
                'charts_generated': list(visualizations.keys()),
                'output_dir': self.output_dir
            }
            
            # Save metadata locally
            metadata_filename = f"metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            metadata_path = os.path.join(self.output_dir, metadata_filename)
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return visualizations
            
        except Exception as e:
            print(f"Error generating comprehensive visualizations: {e}")
            return {}
    
    def list_available_charts(self) -> List[Dict]:
        """List all available charts in local directory"""
        try:
            charts = []
            if os.path.exists(self.output_dir):
                for filename in os.listdir(self.output_dir):
                    if filename.endswith('.png'):
                        filepath = os.path.join(self.output_dir, filename)
                        stat = os.stat(filepath)
                        charts.append({
                            'key': filename,
                            'path': filepath,
                            'last_modified': datetime.fromtimestamp(stat.st_mtime),
                            'size': stat.st_size
                        })
            
            return sorted(charts, key=lambda x: x['last_modified'], reverse=True)
            
        except Exception as e:
            print(f"Error listing charts: {e}")
            return [] 