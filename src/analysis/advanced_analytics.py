#!/usr/bin/env python3
"""
Advanced Analytics Module for FRED Economic Data
Performs comprehensive statistical analysis, modeling, and insights extraction.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose

warnings.filterwarnings("ignore")


class AdvancedAnalytics:
    """
    Comprehensive analytics class for FRED economic data.
    Performs EDA, statistical modeling, segmentation, and time series analysis.
    """

    def __init__(self, data_path=None, df=None):
        """Initialize with data path or DataFrame."""
        if df is not None:
            self.df = df
        elif data_path:
            self.df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        else:
            raise ValueError("Must provide either data_path or DataFrame")

        self.scaler = StandardScaler()
        self.results = {}

    def perform_eda(self):
        """Perform comprehensive Exploratory Data Analysis."""
        print("=" * 60)
        print("EXPLORATORY DATA ANALYSIS")
        print("=" * 60)

        # Basic info
        print(f"\nDataset Shape: {self.df.shape}")
        print(f"Date Range: {self.df.index.min()} to {self.df.index.max()}")
        print(f"Variables: {list(self.df.columns)}")

        # Descriptive statistics
        print("\n" + "=" * 40)
        print("DESCRIPTIVE STATISTICS")
        print("=" * 40)
        desc_stats = self.df.describe()
        print(desc_stats)

        # Skewness and Kurtosis
        print("\n" + "=" * 40)
        print("SKEWNESS AND KURTOSIS")
        print("=" * 40)
        skewness = self.df.skew()
        kurtosis = self.df.kurtosis()

        for col in self.df.columns:
            print(f"{col}:")
            print(f"  Skewness: {skewness[col]:.3f}")
            print(f"  Kurtosis: {kurtosis[col]:.3f}")

        # Correlation Analysis
        print("\n" + "=" * 40)
        print("CORRELATION ANALYSIS")
        print("=" * 40)

        # Pearson correlation
        pearson_corr = self.df.corr(method="pearson")
        print("\nPearson Correlation Matrix:")
        print(pearson_corr.round(3))

        # Spearman correlation
        spearman_corr = self.df.corr(method="spearman")
        print("\nSpearman Correlation Matrix:")
        print(spearman_corr.round(3))

        # Store results
        self.results["eda"] = {
            "descriptive_stats": desc_stats,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "pearson_corr": pearson_corr,
            "spearman_corr": spearman_corr,
        }

        return self.results["eda"]

    def perform_dimensionality_reduction(self, method="pca", n_components=2):
        """Perform dimensionality reduction for visualization."""
        print("\n" + "=" * 40)
        print(f"DIMENSIONALITY REDUCTION ({method.upper()})")
        print("=" * 40)

        # Prepare data (remove NaN values)
        df_clean = self.df.dropna()

        if method.lower() == "pca":
            # PCA
            pca = PCA(n_components=n_components)
            scaled_data = self.scaler.fit_transform(df_clean)
            pca_result = pca.fit_transform(scaled_data)

            print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
            print(f"Total explained variance: {sum(pca.explained_variance_ratio_):.3f}")

            # Create DataFrame with PCA results
            pca_df = pd.DataFrame(
                pca_result,
                columns=[f"PC{i+1}" for i in range(n_components)],
                index=df_clean.index,
            )

            self.results["pca"] = {
                "components": pca_df,
                "explained_variance": pca.explained_variance_ratio_,
                "feature_importance": pd.DataFrame(
                    pca.components_.T,
                    columns=[f"PC{i+1}" for i in range(n_components)],
                    index=df_clean.columns,
                ),
            }

            return self.results["pca"]

        return None

    def perform_statistical_modeling(self, target_var="GDP", test_size=0.2):
        """Perform linear regression with comprehensive diagnostics."""
        print("\n" + "=" * 40)
        print("STATISTICAL MODELING - LINEAR REGRESSION")
        print("=" * 40)

        # Prepare data
        df_clean = self.df.dropna()

        if target_var not in df_clean.columns:
            print(f"Target variable '{target_var}' not found in dataset")
            return None

        # Prepare features and target
        feature_cols = [col for col in df_clean.columns if col != target_var]
        X = df_clean[feature_cols]
        y = df_clean[target_var]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Fit linear regression
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Model performance
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

        print(f"\nModel Performance:")
        print(f"R² (Training): {r2_train:.4f}")
        print(f"R² (Test): {r2_test:.4f}")
        print(f"RMSE (Training): {rmse_train:.4f}")
        print(f"RMSE (Test): {rmse_test:.4f}")

        # Coefficients
        print(f"\nCoefficients:")
        for feature, coef in zip(feature_cols, model.coef_):
            print(f"  {feature}: {coef:.4f}")
        print(f"  Intercept: {model.intercept_:.4f}")

        # Statistical significance using statsmodels
        X_with_const = sm.add_constant(X_train)
        model_sm = sm.OLS(y_train, X_with_const).fit()

        print(f"\nStatistical Significance:")
        print(model_sm.summary().tables[1])

        # Assumption tests
        print(f"\n" + "=" * 30)
        print("REGRESSION ASSUMPTIONS")
        print("=" * 30)

        # 1. Normality of residuals
        residuals = y_train - y_pred_train
        _, p_value_norm = stats.normaltest(residuals)
        print(f"Normality test (p-value): {p_value_norm:.4f}")

        # 2. Multicollinearity (VIF)
        vif_data = []
        for i in range(X_train.shape[1]):
            try:
                vif = variance_inflation_factor(X_train.values, i)
                vif_data.append(vif)
            except:
                vif_data.append(np.nan)

        print(f"\nVariance Inflation Factors:")
        for feature, vif in zip(feature_cols, vif_data):
            print(f"  {feature}: {vif:.3f}")

        # 3. Homoscedasticity
        try:
            _, p_value_het = het_breuschpagan(residuals, X_with_const)
            print(f"\nHomoscedasticity test (p-value): {p_value_het:.4f}")
        except:
            p_value_het = np.nan
            print(f"\nHomoscedasticity test failed")

        # Store results
        self.results["regression"] = {
            "model": model,
            "model_sm": model_sm,
            "performance": {
                "r2_train": r2_train,
                "r2_test": r2_test,
                "rmse_train": rmse_train,
                "rmse_test": rmse_test,
            },
            "coefficients": dict(zip(feature_cols, model.coef_)),
            "assumptions": {
                "normality_p": p_value_norm,
                "homoscedasticity_p": p_value_het,
                "vif": dict(zip(feature_cols, vif_data)),
            },
        }

        return self.results["regression"]

    def perform_clustering(self, max_k=10):
        """Perform clustering analysis with optimal k selection."""
        print("\n" + "=" * 40)
        print("CLUSTERING ANALYSIS")
        print("=" * 40)

        # Prepare data
        df_clean = self.df.dropna()
        if df_clean.shape[0] < 10 or df_clean.shape[1] < 2:
            print(
                "Not enough data for clustering (need at least 10 rows and 2 columns after dropna). Skipping."
            )
            self.results["clustering"] = None
            return None
        try:
            scaled_data = self.scaler.fit_transform(df_clean)
        except Exception as e:
            print(f"Scaling failed: {e}")
            self.results["clustering"] = None
            return None
        # Find optimal k using elbow method and silhouette score
        inertias = []
        silhouette_scores = []
        k_range = range(2, min(max_k + 1, len(df_clean) // 10 + 1))
        if len(k_range) < 2:
            print("Not enough data for multiple clusters. Skipping clustering.")
            self.results["clustering"] = None
            return None
        try:
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(scaled_data)
                inertias.append(kmeans.inertia_)
                silhouette_scores.append(silhouette_score(scaled_data, kmeans.labels_))
            # Plot elbow curve only if there are results
            if inertias and silhouette_scores:
                plt.figure(figsize=(12, 4))
                plt.subplot(1, 2, 1)
                plt.plot(list(k_range), inertias, "bo-")
                plt.xlabel("Number of Clusters (k)")
                plt.ylabel("Inertia")
                plt.title("Elbow Method")
                plt.grid(True)
                plt.subplot(1, 2, 2)
                plt.plot(list(k_range), silhouette_scores, "ro-")
                plt.xlabel("Number of Clusters (k)")
                plt.ylabel("Silhouette Score")
                plt.title("Silhouette Analysis")
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(
                    "data/exports/clustering_analysis.png", dpi=300, bbox_inches="tight"
                )
                plt.show()
            # Choose optimal k (highest silhouette score)
            optimal_k = list(k_range)[np.argmax(silhouette_scores)]
            print(f"Optimal number of clusters: {optimal_k}")
            print(f"Best silhouette score: {max(silhouette_scores):.3f}")
            # Perform clustering with optimal k
            kmeans_optimal = KMeans(n_clusters=optimal_k, random_state=42)
            cluster_labels = kmeans_optimal.fit_predict(scaled_data)
            # Add cluster labels to data
            df_clustered = df_clean.copy()
            df_clustered["Cluster"] = cluster_labels
            # Cluster characteristics
            print(f"\nCluster Characteristics:")
            cluster_stats = df_clustered.groupby("Cluster").agg(["mean", "std"])
            print(cluster_stats.round(3))
            # Store results
            self.results["clustering"] = {
                "optimal_k": optimal_k,
                "silhouette_score": max(silhouette_scores),
                "cluster_labels": cluster_labels,
                "clustered_data": df_clustered,
                "cluster_stats": cluster_stats,
                "inertias": inertias,
                "silhouette_scores": silhouette_scores,
            }
            return self.results["clustering"]
        except Exception as e:
            print(f"Clustering failed: {e}")
            self.results["clustering"] = None
            return None

    def perform_time_series_analysis(self, target_var="GDP"):
        """Perform comprehensive time series analysis."""
        print("\n" + "=" * 40)
        print("TIME SERIES ANALYSIS")
        print("=" * 40)

        if target_var not in self.df.columns:
            print(f"Target variable '{target_var}' not found")
            self.results["time_series"] = None
            return None
        # Prepare time series data
        ts_data = self.df[target_var].dropna()
        if len(ts_data) < 50:
            print(
                "Insufficient data for time series analysis (need at least 50 points). Skipping."
            )
            self.results["time_series"] = None
            return None
        print(f"Time series length: {len(ts_data)} observations")
        print(f"Date range: {ts_data.index.min()} to {ts_data.index.max()}")
        # 1. Time Series Decomposition
        print(f"\nTime Series Decomposition:")
        try:
            # Resample to monthly data if needed
            if ts_data.index.freq is None:
                ts_monthly = ts_data.resample("M").mean()
            else:
                ts_monthly = ts_data
            decomposition = seasonal_decompose(ts_monthly, model="additive", period=12)
            # Plot decomposition
            fig, axes = plt.subplots(4, 1, figsize=(12, 10))
            decomposition.observed.plot(ax=axes[0], title="Original Time Series")
            decomposition.trend.plot(ax=axes[1], title="Trend")
            decomposition.seasonal.plot(ax=axes[2], title="Seasonality")
            decomposition.resid.plot(ax=axes[3], title="Residuals")
            plt.tight_layout()
            plt.savefig(
                "data/exports/time_series_decomposition.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.show()
        except Exception as e:
            print(f"Decomposition failed: {e}")
        # 2. ARIMA Modeling
        print(f"\nARIMA Modeling:")
        try:
            # Fit ARIMA model
            model = ARIMA(ts_monthly, order=(1, 1, 1))
            fitted_model = model.fit()
            print(f"ARIMA Model Summary:")
            print(fitted_model.summary())
            # Forecast
            forecast_steps = min(12, len(ts_monthly) // 4)
            forecast = fitted_model.forecast(steps=forecast_steps)
            conf_int = fitted_model.get_forecast(steps=forecast_steps).conf_int()
            # Plot forecast
            plt.figure(figsize=(12, 6))
            ts_monthly.plot(label="Historical Data")
            forecast.plot(label="Forecast", color="red")
            plt.fill_between(
                forecast.index,
                conf_int.iloc[:, 0],
                conf_int.iloc[:, 1],
                alpha=0.3,
                color="red",
                label="Confidence Interval",
            )
            plt.title(f"{target_var} - ARIMA Forecast")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(
                "data/exports/time_series_forecast.png", dpi=300, bbox_inches="tight"
            )
            plt.show()
            # Store results
            self.results["time_series"] = {
                "model": fitted_model,
                "forecast": forecast,
                "confidence_intervals": conf_int,
                "decomposition": decomposition if "decomposition" in locals() else None,
            }
        except Exception as e:
            print(f"ARIMA modeling failed: {e}")
            self.results["time_series"] = None
        return self.results.get("time_series")

    def generate_insights_report(self):
        """Generate comprehensive insights report in layman's terms."""
        print("\n" + "=" * 60)
        print("COMPREHENSIVE INSIGHTS REPORT")
        print("=" * 60)

        insights = []
        # EDA Insights
        if "eda" in self.results and self.results["eda"] is not None:
            insights.append("EXPLORATORY DATA ANALYSIS INSIGHTS:")
            insights.append("-" * 40)
            # Correlation insights
            pearson_corr = self.results["eda"]["pearson_corr"]
            high_corr_pairs = []
            for i in range(len(pearson_corr.columns)):
                for j in range(i + 1, len(pearson_corr.columns)):
                    corr_val = pearson_corr.iloc[i, j]
                    if abs(corr_val) > 0.7:
                        high_corr_pairs.append(
                            (pearson_corr.columns[i], pearson_corr.columns[j], corr_val)
                        )
            if high_corr_pairs:
                insights.append("Strong correlations found:")
                for var1, var2, corr in high_corr_pairs:
                    insights.append(f"  • {var1} and {var2}: {corr:.3f}")
            else:
                insights.append(
                    "No strong correlations (>0.7) found between variables."
                )
        else:
            insights.append("EDA could not be performed or returned no results.")
        # Regression Insights
        if "regression" in self.results and self.results["regression"] is not None:
            insights.append("\nREGRESSION MODEL INSIGHTS:")
            insights.append("-" * 40)
            reg_results = self.results["regression"]
            r2_test = reg_results["performance"]["r2_test"]
            insights.append(f"Model Performance:")
            insights.append(
                f"  • The model explains {r2_test:.1%} of the variation in the target variable"
            )
            if r2_test > 0.7:
                insights.append("  • This is considered a good model fit")
            elif r2_test > 0.5:
                insights.append("  • This is considered a moderate model fit")
            else:
                insights.append("  • This model has limited predictive power")
            # Assumption insights
            assumptions = reg_results["assumptions"]
            if assumptions["normality_p"] > 0.05:
                insights.append(
                    "  • Residuals are normally distributed (assumption met)"
                )
            else:
                insights.append(
                    "  • Residuals are not normally distributed (assumption violated)"
                )
        else:
            insights.append(
                "Regression modeling could not be performed or returned no results."
            )
        # Clustering Insights
        if "clustering" in self.results and self.results["clustering"] is not None:
            insights.append("\nCLUSTERING INSIGHTS:")
            insights.append("-" * 40)
            cluster_results = self.results["clustering"]
            optimal_k = cluster_results["optimal_k"]
            silhouette_score = cluster_results["silhouette_score"]
            insights.append(f"Optimal number of clusters: {optimal_k}")
            insights.append(f"Cluster quality score: {silhouette_score:.3f}")
            if silhouette_score > 0.5:
                insights.append("  • Clusters are well-separated and distinct")
            elif silhouette_score > 0.3:
                insights.append("  • Clusters show moderate separation")
            else:
                insights.append("  • Clusters may not be well-defined")
        else:
            insights.append("Clustering could not be performed or returned no results.")
        # Time Series Insights
        if "time_series" in self.results and self.results["time_series"] is not None:
            insights.append("\nTIME SERIES INSIGHTS:")
            insights.append("-" * 40)
            insights.append(
                "  • Time series decomposition shows trend, seasonality, and random components"
            )
            insights.append(
                "  • ARIMA model provides future forecasts with confidence intervals"
            )
            insights.append(
                "  • Forecasts can be used for planning and decision-making"
            )
        else:
            insights.append(
                "Time series analysis could not be performed or returned no results."
            )
        # Print insights
        for insight in insights:
            print(insight)
        # Save insights to file
        with open("data/exports/insights_report.txt", "w") as f:
            f.write("\n".join(insights))
        return insights

    def run_complete_analysis(self):
        """Run the complete advanced analytics workflow."""
        print("Starting comprehensive advanced analytics...")

        # 1. EDA
        self.perform_eda()

        # 2. Dimensionality reduction
        self.perform_dimensionality_reduction()

        # 3. Statistical modeling
        self.perform_statistical_modeling()

        # 4. Clustering
        self.perform_clustering()

        # 5. Time series analysis
        self.perform_time_series_analysis()

        # 6. Generate insights
        self.generate_insights_report()

        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE!")
        print("=" * 60)
        print("Check the following outputs:")
        print("  • data/exports/insights_report.txt - Comprehensive insights")
        print("  • data/exports/clustering_analysis.png - Clustering results")
        print(
            "  • data/exports/time_series_decomposition.png - Time series decomposition"
        )
        print("  • data/exports/time_series_forecast.png - Time series forecast")

        return self.results
