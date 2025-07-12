#!/usr/bin/env python3
"""
Chart Generator for FRED ML
Creates comprehensive economic visualizations and stores them in S3
"""

import io
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import boto3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Use hardcoded defaults to avoid import issues
DEFAULT_REGION = 'us-east-1'

# Set style for matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class ChartGenerator:
    """Generate comprehensive economic visualizations"""
    
    def __init__(self, s3_bucket: str = 'fredmlv1', aws_region: str = None):
        self.s3_bucket = s3_bucket
        if aws_region is None:
            aws_region = DEFAULT_REGION
        self.s3_client = boto3.client('s3', region_name=aws_region)
        self.chart_paths = []
        
    def create_time_series_chart(self, df: pd.DataFrame, title: str = "Economic Indicators") -> str:
        """Create time series chart and upload to S3"""
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
            
            # Save to bytes
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
            img_buffer.seek(0)
            
            # Upload to S3
            chart_key = f"visualizations/time_series_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=chart_key,
                Body=img_buffer.getvalue(),
                ContentType='image/png'
            )
            
            plt.close()
            self.chart_paths.append(chart_key)
            return chart_key
            
        except Exception as e:
            print(f"Error creating time series chart: {e}")
            return None
    
    def create_correlation_heatmap(self, df: pd.DataFrame) -> str:
        """Create correlation heatmap and upload to S3"""
        try:
            corr_matrix = df.corr()
            
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                       square=True, linewidths=0.5, cbar_kws={"shrink": .8})
            
            plt.title('Economic Indicators Correlation Matrix', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # Save to bytes
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
            img_buffer.seek(0)
            
            # Upload to S3
            chart_key = f"visualizations/correlation_heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=chart_key,
                Body=img_buffer.getvalue(),
                ContentType='image/png'
            )
            
            plt.close()
            self.chart_paths.append(chart_key)
            return chart_key
            
        except Exception as e:
            print(f"Error creating correlation heatmap: {e}")
            return None
    
    def create_distribution_charts(self, df: pd.DataFrame) -> List[str]:
        """Create distribution charts for each indicator"""
        chart_keys = []
        
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
                    
                    # Save to bytes
                    img_buffer = io.BytesIO()
                    plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
                    img_buffer.seek(0)
                    
                    # Upload to S3
                    chart_key = f"visualizations/distribution_{column}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    self.s3_client.put_object(
                        Bucket=self.s3_bucket,
                        Key=chart_key,
                        Body=img_buffer.getvalue(),
                        ContentType='image/png'
                    )
                    
                    plt.close()
                    chart_keys.append(chart_key)
                    self.chart_paths.append(chart_key)
            
            return chart_keys
            
        except Exception as e:
            print(f"Error creating distribution charts: {e}")
            return []
    
    def create_pca_visualization(self, df: pd.DataFrame, n_components: int = 2) -> str:
        """Create PCA visualization and upload to S3"""
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
            
            # Save to bytes
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
            img_buffer.seek(0)
            
            # Upload to S3
            chart_key = f"visualizations/pca_visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=chart_key,
                Body=img_buffer.getvalue(),
                ContentType='image/png'
            )
            
            plt.close()
            self.chart_paths.append(chart_key)
            return chart_key
            
        except Exception as e:
            print(f"Error creating PCA visualization: {e}")
            return None
    
    def create_forecast_chart(self, historical_data: pd.Series, forecast_data: List[float], 
                            title: str = "Economic Forecast") -> str:
        """Create forecast chart and upload to S3"""
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
            
            # Save to bytes
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
            img_buffer.seek(0)
            
            # Upload to S3
            chart_key = f"visualizations/forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=chart_key,
                Body=img_buffer.getvalue(),
                ContentType='image/png'
            )
            
            plt.close()
            self.chart_paths.append(chart_key)
            return chart_key
            
        except Exception as e:
            print(f"Error creating forecast chart: {e}")
            return None
    
    def create_regression_diagnostics(self, y_true: List[float], y_pred: List[float], 
                                   residuals: List[float]) -> str:
        """Create regression diagnostics chart and upload to S3"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Actual vs Predicted
            axes[0, 0].scatter(y_true, y_pred, alpha=0.6)
            axes[0, 0].plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', lw=2)
            axes[0, 0].set_xlabel('Actual Values')
            axes[0, 0].set_ylabel('Predicted Values')
            axes[0, 0].set_title('Actual vs Predicted')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Residuals vs Predicted
            axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
            axes[0, 1].axhline(y=0, color='r', linestyle='--')
            axes[0, 1].set_xlabel('Predicted Values')
            axes[0, 1].set_ylabel('Residuals')
            axes[0, 1].set_title('Residuals vs Predicted')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Residuals histogram
            axes[1, 0].hist(residuals, bins=20, alpha=0.7, edgecolor='black')
            axes[1, 0].set_xlabel('Residuals')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Residuals Distribution')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Q-Q plot
            from scipy import stats
            stats.probplot(residuals, dist="norm", plot=axes[1, 1])
            axes[1, 1].set_title('Q-Q Plot of Residuals')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save to bytes
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
            img_buffer.seek(0)
            
            # Upload to S3
            chart_key = f"visualizations/regression_diagnostics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=chart_key,
                Body=img_buffer.getvalue(),
                ContentType='image/png'
            )
            
            plt.close()
            self.chart_paths.append(chart_key)
            return chart_key
            
        except Exception as e:
            print(f"Error creating regression diagnostics: {e}")
            return None
    
    def create_clustering_chart(self, df: pd.DataFrame, n_clusters: int = 3) -> str:
        """Create clustering visualization and upload to S3"""
        try:
            from sklearn.cluster import KMeans
            
            # Prepare data
            df_clean = df.dropna()
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
            
            # Save to bytes
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
            img_buffer.seek(0)
            
            # Upload to S3
            chart_key = f"visualizations/clustering_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=chart_key,
                Body=img_buffer.getvalue(),
                ContentType='image/png'
            )
            
            plt.close()
            self.chart_paths.append(chart_key)
            return chart_key
            
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
                's3_bucket': self.s3_bucket
            }
            
            # Upload metadata
            metadata_key = f"visualizations/metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=metadata_key,
                Body=json.dumps(metadata, indent=2),
                ContentType='application/json'
            )
            
            return visualizations
            
        except Exception as e:
            print(f"Error generating comprehensive visualizations: {e}")
            return {}
    
    def get_chart_url(self, chart_key: str) -> str:
        """Get public URL for a chart"""
        try:
            return f"https://{self.s3_bucket}.s3.amazonaws.com/{chart_key}"
        except Exception as e:
            print(f"Error generating chart URL: {e}")
            return None
    
    def list_available_charts(self) -> List[Dict]:
        """List all available charts in S3"""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.s3_bucket,
                Prefix='visualizations/'
            )
            
            charts = []
            if 'Contents' in response:
                for obj in response['Contents']:
                    if obj['Key'].endswith('.png'):
                        charts.append({
                            'key': obj['Key'],
                            'last_modified': obj['LastModified'],
                            'size': obj['Size'],
                            'url': self.get_chart_url(obj['Key'])
                        })
            
            return sorted(charts, key=lambda x: x['last_modified'], reverse=True)
            
        except Exception as e:
            print(f"Error listing charts: {e}")
            return [] 