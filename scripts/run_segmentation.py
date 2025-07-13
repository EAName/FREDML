#!/usr/bin/env python
"""
Run Segmentation: K-means clustering, elbow method, silhouette score, PCA visualization
"""
import os
import sys
import glob
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def find_latest_data():
    data_files = glob.glob('data/processed/fred_data_*.csv')
    if not data_files:
        raise FileNotFoundError("No FRED data files found. Run the pipeline first.")
    return max(data_files, key=os.path.getctime)

def main():
    print("="*60)
    print("FRED Segmentation: K-means, Elbow, Silhouette, PCA Visualization")
    print("="*60)
    data_file = find_latest_data()
    print(f"Using data file: {data_file}")
    df = pd.read_csv(data_file, index_col=0, parse_dates=True)
    df_clean = df.dropna()
    if df_clean.shape[0] < 10 or df_clean.shape[1] < 2:
        print("Not enough data for clustering (need at least 10 rows and 2 columns after dropna). Skipping.")
        return
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_clean)
    # Elbow and silhouette
    inertias = []
    silhouette_scores = []
    k_range = range(2, min(11, len(df_clean)//10+1))
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(scaled_data)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(scaled_data, kmeans.labels_))
    # Plot elbow and silhouette
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(list(k_range), inertias, 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    plt.grid(True)
    plt.subplot(1,2,2)
    plt.plot(list(k_range), silhouette_scores, 'ro-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Analysis')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('data/exports/clustering_analysis.png', dpi=200)
    plt.close()
    # Choose optimal k
    optimal_k = list(k_range)[np.argmax(silhouette_scores)]
    print(f"Optimal number of clusters: {optimal_k}")
    print(f"Best silhouette score: {max(silhouette_scores):.3f}")
    # Final clustering
    kmeans_optimal = KMeans(n_clusters=optimal_k, random_state=42)
    cluster_labels = kmeans_optimal.fit_predict(scaled_data)
    df_clustered = df_clean.copy()
    df_clustered['Cluster'] = cluster_labels
    # Cluster stats
    cluster_stats = df_clustered.groupby('Cluster').agg(['mean', 'std'])
    print("\nCluster Characteristics:")
    print(cluster_stats.round(3))
    # PCA visualization
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(pca_result[:,0], pca_result[:,1], c=cluster_labels, cmap='tab10', alpha=0.7)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Clusters Visualized with PCA')
    plt.legend(*scatter.legend_elements(), title="Cluster")
    plt.tight_layout()
    plt.savefig('data/exports/clusters_pca.png', dpi=200)
    plt.close()
    print("\nSegmentation complete. Outputs saved to data/exports/.")

if __name__ == "__main__":
    main() 