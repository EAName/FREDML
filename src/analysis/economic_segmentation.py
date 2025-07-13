"""
Economic Segmentation Module
Advanced clustering analysis for economic time series and time periods
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform

logger = logging.getLogger(__name__)

class EconomicSegmentation:
    """
    Advanced economic segmentation using clustering techniques
    for both time periods and economic series
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize segmentation with economic data
        
        Args:
            data: DataFrame with economic indicators
        """
        self.data = data.copy()
        self.scaler = StandardScaler()
        self.clusters = {}
        self.cluster_analysis = {}
        
    def prepare_time_period_data(self, indicators: List[str] = None, 
                                window_size: int = 4) -> pd.DataFrame:
        """
        Prepare time period data for clustering
        
        Args:
            indicators: List of indicators to use. If None, use all numeric columns
            window_size: Rolling window size for feature extraction
            
        Returns:
            DataFrame with time period features
        """
        if indicators is None:
            indicators = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Calculate growth rates for economic indicators
        growth_data = self.data[indicators].pct_change().dropna()
        
        # Extract features for each time period
        features = []
        feature_names = []
        
        for indicator in indicators:
            # Rolling statistics
            features.extend([
                growth_data[indicator].rolling(window_size).mean(),
                growth_data[indicator].rolling(window_size).std(),
                growth_data[indicator].rolling(window_size).min(),
                growth_data[indicator].rolling(window_size).max(),
                growth_data[indicator].rolling(window_size).skew(),
                growth_data[indicator].rolling(window_size).kurt()
            ])
            feature_names.extend([
                f"{indicator}_mean", f"{indicator}_std", f"{indicator}_min",
                f"{indicator}_max", f"{indicator}_skew", f"{indicator}_kurt"
            ])
        
        # Create feature matrix
        feature_df = pd.concat(features, axis=1)
        feature_df.columns = feature_names
        feature_df = feature_df.dropna()
        
        return feature_df
    
    def prepare_series_data(self, indicators: List[str] = None) -> pd.DataFrame:
        """
        Prepare series data for clustering (clustering the indicators themselves)
        
        Args:
            indicators: List of indicators to use. If None, use all numeric columns
            
        Returns:
            DataFrame with series features
        """
        if indicators is None:
            indicators = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Calculate growth rates
        growth_data = self.data[indicators].pct_change().dropna()
        
        # Extract features for each series
        series_features = {}
        
        for indicator in indicators:
            series = growth_data[indicator].dropna()
            
            # Statistical features
            series_features[indicator] = {
                'mean': series.mean(),
                'std': series.std(),
                'min': series.min(),
                'max': series.max(),
                'skew': series.skew(),
                'kurt': series.kurtosis(),
                'autocorr_1': series.autocorr(lag=1),
                'autocorr_4': series.autocorr(lag=4),
                'volatility': series.rolling(12).std().mean(),
                'trend': np.polyfit(range(len(series)), series, 1)[0]
            }
        
        return pd.DataFrame(series_features).T
    
    def find_optimal_clusters(self, data: pd.DataFrame, max_clusters: int = 10,
                             method: str = 'kmeans') -> Dict:
        """
        Find optimal number of clusters using elbow method and silhouette analysis
        
        Args:
            data: Feature data for clustering
            max_clusters: Maximum number of clusters to test
            method: Clustering method ('kmeans' or 'hierarchical')
            
        Returns:
            Dictionary with optimal cluster analysis
        """
        if len(data) < max_clusters:
            max_clusters = len(data) - 1
        
        inertias = []
        silhouette_scores = []
        calinski_scores = []
        
        for k in range(2, max_clusters + 1):
            try:
                if method == 'kmeans':
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(data)
                    inertias.append(kmeans.inertia_)
                else:
                    clustering = AgglomerativeClustering(n_clusters=k)
                    labels = clustering.fit_predict(data)
                    inertias.append(0)  # Not applicable for hierarchical
                
                # Calculate scores
                if len(np.unique(labels)) > 1:
                    silhouette_scores.append(silhouette_score(data, labels))
                    calinski_scores.append(calinski_harabasz_score(data, labels))
                else:
                    silhouette_scores.append(0)
                    calinski_scores.append(0)
                    
            except Exception as e:
                logger.warning(f"Failed to cluster with k={k}: {e}")
                inertias.append(0)
                silhouette_scores.append(0)
                calinski_scores.append(0)
        
        # Find optimal k using silhouette score
        optimal_k_silhouette = np.argmax(silhouette_scores) + 2
        optimal_k_calinski = np.argmax(calinski_scores) + 2
        
        # Elbow method (for k-means)
        if method == 'kmeans' and len(inertias) > 1:
            # Calculate second derivative to find elbow
            second_derivative = np.diff(np.diff(inertias))
            optimal_k_elbow = np.argmin(second_derivative) + 3
        else:
            optimal_k_elbow = optimal_k_silhouette
        
        return {
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'calinski_scores': calinski_scores,
            'optimal_k_silhouette': optimal_k_silhouette,
            'optimal_k_calinski': optimal_k_calinski,
            'optimal_k_elbow': optimal_k_elbow,
            'recommended_k': optimal_k_silhouette  # Use silhouette as primary
        }
    
    def cluster_time_periods(self, indicators: List[str] = None, 
                           n_clusters: int = None, method: str = 'kmeans',
                           window_size: int = 4) -> Dict:
        """
        Cluster time periods based on economic activity patterns
        
        Args:
            indicators: List of indicators to use
            n_clusters: Number of clusters. If None, auto-detect
            method: Clustering method ('kmeans' or 'hierarchical')
            window_size: Rolling window size for feature extraction
            
        Returns:
            Dictionary with clustering results
        """
        # Prepare data
        feature_df = self.prepare_time_period_data(indicators, window_size)
        
        # Scale features
        scaled_data = self.scaler.fit_transform(feature_df)
        scaled_df = pd.DataFrame(scaled_data, index=feature_df.index, columns=feature_df.columns)
        
        # Find optimal clusters if not specified
        if n_clusters is None:
            cluster_analysis = self.find_optimal_clusters(scaled_df, method=method)
            n_clusters = cluster_analysis['recommended_k']
            logger.info(f"Auto-detected optimal clusters: {n_clusters}")
        
        # Perform clustering
        if method == 'kmeans':
            clustering = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        else:
            clustering = AgglomerativeClustering(n_clusters=n_clusters)
        
        cluster_labels = clustering.fit_predict(scaled_df)
        
        # Add cluster labels to original data
        result_df = feature_df.copy()
        result_df['cluster'] = cluster_labels
        
        # Analyze clusters
        cluster_analysis = self.analyze_clusters(result_df, 'cluster')
        
        # Dimensionality reduction for visualization
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(scaled_data)
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(scaled_data)-1))
        tsne_data = tsne.fit_transform(scaled_data)
        
        return {
            'data': result_df,
            'cluster_labels': cluster_labels,
            'cluster_analysis': cluster_analysis,
            'pca_data': pca_data,
            'tsne_data': tsne_data,
            'feature_importance': dict(zip(feature_df.columns, np.abs(pca.components_[0]))),
            'n_clusters': n_clusters,
            'method': method
        }
    
    def cluster_economic_series(self, indicators: List[str] = None,
                              n_clusters: int = None, method: str = 'kmeans') -> Dict:
        """
        Cluster economic series based on their characteristics
        
        Args:
            indicators: List of indicators to use
            n_clusters: Number of clusters. If None, auto-detect
            method: Clustering method ('kmeans' or 'hierarchical')
            
        Returns:
            Dictionary with clustering results
        """
        # Prepare data
        series_df = self.prepare_series_data(indicators)
        
        # Scale features
        scaled_data = self.scaler.fit_transform(series_df)
        scaled_df = pd.DataFrame(scaled_data, index=series_df.index, columns=series_df.columns)
        
        # Find optimal clusters if not specified
        if n_clusters is None:
            cluster_analysis = self.find_optimal_clusters(scaled_df, method=method)
            n_clusters = cluster_analysis['recommended_k']
            logger.info(f"Auto-detected optimal clusters: {n_clusters}")
        
        # Perform clustering
        if method == 'kmeans':
            clustering = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        else:
            clustering = AgglomerativeClustering(n_clusters=n_clusters)
        
        cluster_labels = clustering.fit_predict(scaled_df)
        
        # Add cluster labels
        result_df = series_df.copy()
        result_df['cluster'] = cluster_labels
        
        # Analyze clusters
        cluster_analysis = self.analyze_clusters(result_df, 'cluster')
        
        # Dimensionality reduction for visualization
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(scaled_data)
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(scaled_data)-1))
        tsne_data = tsne.fit_transform(scaled_data)
        
        return {
            'data': result_df,
            'cluster_labels': cluster_labels,
            'cluster_analysis': cluster_analysis,
            'pca_data': pca_data,
            'tsne_data': tsne_data,
            'feature_importance': dict(zip(series_df.columns, np.abs(pca.components_[0]))),
            'n_clusters': n_clusters,
            'method': method
        }
    
    def analyze_clusters(self, data: pd.DataFrame, cluster_col: str) -> Dict:
        """
        Analyze cluster characteristics
        
        Args:
            data: DataFrame with cluster labels
            cluster_col: Name of cluster column
            
        Returns:
            Dictionary with cluster analysis
        """
        feature_cols = [col for col in data.columns if col != cluster_col]
        cluster_analysis = {}
        
        for cluster_id in data[cluster_col].unique():
            cluster_data = data[data[cluster_col] == cluster_id]
            
            cluster_analysis[cluster_id] = {
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(data) * 100,
                'features': {}
            }
            
            # Analyze each feature
            for feature in feature_cols:
                feature_data = cluster_data[feature]
                cluster_analysis[cluster_id]['features'][feature] = {
                    'mean': feature_data.mean(),
                    'std': feature_data.std(),
                    'min': feature_data.min(),
                    'max': feature_data.max(),
                    'median': feature_data.median()
                }
        
        return cluster_analysis
    
    def perform_hierarchical_clustering(self, data: pd.DataFrame, 
                                     method: str = 'ward', 
                                     distance_threshold: float = None) -> Dict:
        """
        Perform hierarchical clustering with dendrogram analysis
        
        Args:
            data: Feature data for clustering
            method: Linkage method ('ward', 'complete', 'average', 'single')
            distance_threshold: Distance threshold for cutting dendrogram
            
        Returns:
            Dictionary with hierarchical clustering results
        """
        # Scale data
        scaled_data = self.scaler.fit_transform(data)
        
        # Calculate linkage matrix
        if method == 'ward':
            linkage_matrix = linkage(scaled_data, method=method)
        else:
            # For non-ward methods, we need to provide distance matrix
            distance_matrix = pdist(scaled_data)
            linkage_matrix = linkage(distance_matrix, method=method)
        
        # Determine number of clusters
        if distance_threshold is None:
            # Use elbow method on distance
            distances = linkage_matrix[:, 2]
            second_derivative = np.diff(np.diff(distances))
            optimal_threshold = distances[np.argmax(second_derivative) + 1]
        else:
            optimal_threshold = distance_threshold
        
        # Get cluster labels
        cluster_labels = fcluster(linkage_matrix, optimal_threshold, criterion='distance')
        
        # Analyze clusters
        result_df = data.copy()
        result_df['cluster'] = cluster_labels
        cluster_analysis = self.analyze_clusters(result_df, 'cluster')
        
        return {
            'linkage_matrix': linkage_matrix,
            'cluster_labels': cluster_labels,
            'distance_threshold': optimal_threshold,
            'cluster_analysis': cluster_analysis,
            'data': result_df,
            'method': method
        }
    
    def generate_segmentation_report(self, time_period_clusters: Dict = None,
                                   series_clusters: Dict = None) -> str:
        """
        Generate comprehensive segmentation report
        
        Args:
            time_period_clusters: Results from time period clustering
            series_clusters: Results from series clustering
            
        Returns:
            Formatted report string
        """
        report = "ECONOMIC SEGMENTATION REPORT\n"
        report += "=" * 50 + "\n\n"
        
        if time_period_clusters:
            report += "TIME PERIOD CLUSTERING\n"
            report += "-" * 30 + "\n"
            report += f"Method: {time_period_clusters['method']}\n"
            report += f"Number of Clusters: {time_period_clusters['n_clusters']}\n"
            report += f"Total Periods: {len(time_period_clusters['data'])}\n\n"
            
            # Cluster summary
            cluster_analysis = time_period_clusters['cluster_analysis']
            for cluster_id, analysis in cluster_analysis.items():
                report += f"Cluster {cluster_id}:\n"
                report += f"  Size: {analysis['size']} periods ({analysis['percentage']:.1f}%)\n"
                
                # Top features for this cluster
                if 'feature_importance' in time_period_clusters:
                    features = time_period_clusters['feature_importance']
                    top_features = sorted(features.items(), key=lambda x: x[1], reverse=True)[:5]
                    report += f"  Top Features: {', '.join([f[0] for f in top_features])}\n"
                
                report += "\n"
        
        if series_clusters:
            report += "ECONOMIC SERIES CLUSTERING\n"
            report += "-" * 30 + "\n"
            report += f"Method: {series_clusters['method']}\n"
            report += f"Number of Clusters: {series_clusters['n_clusters']}\n"
            report += f"Total Series: {len(series_clusters['data'])}\n\n"
            
            # Cluster summary
            cluster_analysis = series_clusters['cluster_analysis']
            for cluster_id, analysis in cluster_analysis.items():
                report += f"Cluster {cluster_id}:\n"
                report += f"  Size: {analysis['size']} series ({analysis['percentage']:.1f}%)\n"
                
                # Series in this cluster
                cluster_series = series_clusters['data'][series_clusters['data']['cluster'] == cluster_id]
                series_names = cluster_series.index.tolist()
                report += f"  Series: {', '.join(series_names)}\n"
                
                # Top features for this cluster
                if 'feature_importance' in series_clusters:
                    features = series_clusters['feature_importance']
                    top_features = sorted(features.items(), key=lambda x: x[1], reverse=True)[:5]
                    report += f"  Top Features: {', '.join([f[0] for f in top_features])}\n"
                
                report += "\n"
        
        return report 