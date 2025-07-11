#!/usr/bin/env python
"""
Run EDA: Distributions, skewness, kurtosis, correlations, PCA/t-SNE
"""
import os
import sys
import glob
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Find latest processed data file
def find_latest_data():
    data_files = glob.glob('data/processed/fred_data_*.csv')
    if not data_files:
        raise FileNotFoundError("No FRED data files found. Run the pipeline first.")
    return max(data_files, key=os.path.getctime)

def main():
    print("="*60)
    print("FRED EDA: Distributions, Skewness, Kurtosis, Correlations, PCA")
    print("="*60)
    data_file = find_latest_data()
    print(f"Using data file: {data_file}")
    df = pd.read_csv(data_file, index_col=0, parse_dates=True)
    df_clean = df.dropna()
    # 1. Distributions, Skewness, Kurtosis
    desc = df.describe()
    skew = df.skew()
    kurt = df.kurtosis()
    print("\nDescriptive Statistics:\n", desc)
    print("\nSkewness:")
    print(skew)
    print("\nKurtosis:")
    print(kurt)
    # Plot distributions
    for col in df.columns:
        plt.figure(figsize=(8,4))
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f"Distribution of {col}")
        plt.savefig(f"data/exports/distribution_{col}.png", dpi=200, bbox_inches='tight')
        plt.close()
    # 2. Correlation matrices
    pearson_corr = df.corr(method='pearson')
    spearman_corr = df.corr(method='spearman')
    print("\nPearson Correlation Matrix:\n", pearson_corr.round(3))
    print("\nSpearman Correlation Matrix:\n", spearman_corr.round(3))
    plt.figure(figsize=(8,6))
    sns.heatmap(pearson_corr, annot=True, cmap='coolwarm', center=0)
    plt.title('Pearson Correlation Matrix')
    plt.tight_layout()
    plt.savefig('data/exports/pearson_corr_matrix.png', dpi=200)
    plt.close()
    plt.figure(figsize=(8,6))
    sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', center=0)
    plt.title('Spearman Correlation Matrix')
    plt.tight_layout()
    plt.savefig('data/exports/spearman_corr_matrix.png', dpi=200)
    plt.close()
    # 3. PCA for visualization
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df_clean)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled)
    pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'], index=df_clean.index)
    plt.figure(figsize=(8,6))
    plt.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.5)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA Projection (2D)')
    plt.tight_layout()
    plt.savefig('data/exports/pca_2d.png', dpi=200)
    plt.close()
    print("\nEDA complete. Outputs saved to data/exports/.")

if __name__ == "__main__":
    main() 