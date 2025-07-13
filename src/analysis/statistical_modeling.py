"""
Statistical Modeling Module
Advanced statistical analysis for economic indicators including regression, correlation, and diagnostics
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import adfuller, kpss

logger = logging.getLogger(__name__)

class StatisticalModeling:
    """
    Advanced statistical modeling for economic indicators
    including regression analysis, correlation analysis, and diagnostic testing
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize statistical modeling with economic data
        
        Args:
            data: DataFrame with economic indicators
        """
        self.data = data.copy()
        self.models = {}
        self.diagnostics = {}
        self.correlations = {}
        
    def prepare_regression_data(self, target: str, predictors: List[str] = None,
                              lag_periods: int = 4) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for regression analysis with lagged variables
        
        Args:
            target: Target variable name
            predictors: List of predictor variables. If None, use all other numeric columns
            lag_periods: Number of lag periods to include
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        if target not in self.data.columns:
            raise ValueError(f"Target variable {target} not found in data")
            
        if predictors is None:
            predictors = [col for col in self.data.select_dtypes(include=[np.number]).columns 
                        if col != target]
        
        # Calculate growth rates for all variables
        growth_data = self.data[[target] + predictors].pct_change().dropna()
        
        # Create lagged features
        feature_data = {}
        
        for predictor in predictors:
            # Current value
            feature_data[predictor] = growth_data[predictor]
            
            # Lagged values
            for lag in range(1, lag_periods + 1):
                feature_data[f"{predictor}_lag{lag}"] = growth_data[predictor].shift(lag)
        
        # Add target variable lags as features
        for lag in range(1, lag_periods + 1):
            feature_data[f"{target}_lag{lag}"] = growth_data[target].shift(lag)
        
        # Create feature matrix
        features_df = pd.DataFrame(feature_data)
        features_df = features_df.dropna()
        
        # Target variable
        target_series = growth_data[target].iloc[features_df.index]
        
        return features_df, target_series
    
    def fit_regression_model(self, target: str, predictors: List[str] = None,
                           lag_periods: int = 4, include_interactions: bool = False) -> Dict:
        """
        Fit linear regression model with diagnostic testing
        
        Args:
            target: Target variable name
            predictors: List of predictor variables
            lag_periods: Number of lag periods to include
            include_interactions: Whether to include interaction terms
            
        Returns:
            Dictionary with model results and diagnostics
        """
        # Prepare data
        features_df, target_series = self.prepare_regression_data(target, predictors, lag_periods)
        
        if include_interactions:
            # Add interaction terms
            interaction_features = []
            feature_cols = features_df.columns.tolist()
            
            for i, col1 in enumerate(feature_cols):
                for col2 in feature_cols[i+1:]:
                    interaction_name = f"{col1}_x_{col2}"
                    interaction_features.append(features_df[col1] * features_df[col2])
                    features_df[interaction_name] = interaction_features[-1]
        
        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_df)
        features_scaled_df = pd.DataFrame(features_scaled, 
                                        index=features_df.index, 
                                        columns=features_df.columns)
        
        # Fit model
        model = LinearRegression()
        model.fit(features_scaled_df, target_series)
        
        # Predictions
        predictions = model.predict(features_scaled_df)
        residuals = target_series - predictions
        
        # Model performance
        r2 = r2_score(target_series, predictions)
        mse = mean_squared_error(target_series, predictions)
        rmse = np.sqrt(mse)
        
        # Coefficient analysis
        coefficients = pd.DataFrame({
            'variable': features_df.columns,
            'coefficient': model.coef_,
            'abs_coefficient': np.abs(model.coef_)
        }).sort_values('abs_coefficient', ascending=False)
        
        # Diagnostic tests
        diagnostics = self.perform_regression_diagnostics(features_scaled_df, target_series, 
                                                        predictions, residuals)
        
        return {
            'model': model,
            'scaler': scaler,
            'features': features_df,
            'target': target_series,
            'predictions': predictions,
            'residuals': residuals,
            'coefficients': coefficients,
            'performance': {
                'r2': r2,
                'mse': mse,
                'rmse': rmse,
                'mae': np.mean(np.abs(residuals))
            },
            'diagnostics': diagnostics
        }
    
    def perform_regression_diagnostics(self, features: pd.DataFrame, target: pd.Series,
                                     predictions: np.ndarray, residuals: pd.Series) -> Dict:
        """
        Perform comprehensive regression diagnostics
        
        Args:
            features: Feature matrix
            target: Target variable
            predictions: Model predictions
            residuals: Model residuals
            
        Returns:
            Dictionary with diagnostic test results
        """
        diagnostics = {}
        
        # 1. Normality test (Shapiro-Wilk)
        try:
            normality_stat, normality_p = stats.shapiro(residuals)
            diagnostics['normality'] = {
                'statistic': normality_stat,
                'p_value': normality_p,
                'is_normal': normality_p > 0.05
            }
        except:
            diagnostics['normality'] = {'error': 'Test failed'}
        
        # 2. Homoscedasticity test (Breusch-Pagan)
        try:
            bp_stat, bp_p, bp_f, bp_f_p = het_breuschpagan(residuals, features)
            diagnostics['homoscedasticity'] = {
                'statistic': bp_stat,
                'p_value': bp_p,
                'f_statistic': bp_f,
                'f_p_value': bp_f_p,
                'is_homoscedastic': bp_p > 0.05
            }
        except:
            diagnostics['homoscedasticity'] = {'error': 'Test failed'}
        
        # 3. Autocorrelation test (Durbin-Watson)
        try:
            dw_stat = durbin_watson(residuals)
            diagnostics['autocorrelation'] = {
                'statistic': dw_stat,
                'interpretation': self._interpret_durbin_watson(dw_stat)
            }
        except:
            diagnostics['autocorrelation'] = {'error': 'Test failed'}
        
        # 4. Multicollinearity test (VIF)
        try:
            vif_scores = {}
            for i, col in enumerate(features.columns):
                vif = variance_inflation_factor(features.values, i)
                vif_scores[col] = vif
            
            diagnostics['multicollinearity'] = {
                'vif_scores': vif_scores,
                'high_vif_variables': [var for var, vif in vif_scores.items() if vif > 10],
                'mean_vif': np.mean(list(vif_scores.values()))
            }
        except:
            diagnostics['multicollinearity'] = {'error': 'Test failed'}
        
        # 5. Stationarity tests
        try:
            # ADF test
            adf_result = adfuller(target)
            diagnostics['stationarity_adf'] = {
                'statistic': adf_result[0],
                'p_value': adf_result[1],
                'is_stationary': adf_result[1] < 0.05
            }
            
            # KPSS test
            kpss_result = kpss(target, regression='c')
            diagnostics['stationarity_kpss'] = {
                'statistic': kpss_result[0],
                'p_value': kpss_result[1],
                'is_stationary': kpss_result[1] > 0.05
            }
        except:
            diagnostics['stationarity'] = {'error': 'Test failed'}
        
        return diagnostics
    
    def _interpret_durbin_watson(self, dw_stat: float) -> str:
        """Interpret Durbin-Watson statistic"""
        if dw_stat < 1.5:
            return "Positive autocorrelation"
        elif dw_stat > 2.5:
            return "Negative autocorrelation"
        else:
            return "No significant autocorrelation"
    
    def analyze_correlations(self, indicators: List[str] = None, 
                           method: str = 'pearson') -> Dict:
        """
        Perform comprehensive correlation analysis
        
        Args:
            indicators: List of indicators to analyze. If None, use all numeric columns
            method: Correlation method ('pearson', 'spearman', 'kendall')
            
        Returns:
            Dictionary with correlation analysis results
        """
        if indicators is None:
            indicators = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Calculate growth rates
        growth_data = self.data[indicators].pct_change().dropna()
        
        # Correlation matrix
        corr_matrix = growth_data.corr(method=method)
        
        # Significant correlations
        significant_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                var1 = corr_matrix.columns[i]
                var2 = corr_matrix.columns[j]
                corr_value = corr_matrix.iloc[i, j]
                
                # Test significance
                n = len(growth_data)
                t_stat = corr_value * np.sqrt((n-2) / (1-corr_value**2))
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n-2))
                
                if p_value < 0.05:
                    significant_correlations.append({
                        'variable1': var1,
                        'variable2': var2,
                        'correlation': corr_value,
                        'p_value': p_value,
                        'strength': self._interpret_correlation_strength(abs(corr_value))
                    })
        
        # Sort by absolute correlation
        significant_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        # Principal Component Analysis
        try:
            pca = self._perform_pca_analysis(growth_data)
        except Exception as e:
            logger.warning(f"PCA analysis failed: {e}")
            pca = {'error': str(e)}
        
        return {
            'correlation_matrix': corr_matrix,
            'significant_correlations': significant_correlations,
            'method': method,
            'pca_analysis': pca
        }
    
    def _interpret_correlation_strength(self, corr_value: float) -> str:
        """Interpret correlation strength"""
        if corr_value >= 0.8:
            return "Very Strong"
        elif corr_value >= 0.6:
            return "Strong"
        elif corr_value >= 0.4:
            return "Moderate"
        elif corr_value >= 0.2:
            return "Weak"
        else:
            return "Very Weak"
    
    def _perform_pca_analysis(self, data: pd.DataFrame) -> Dict:
        """Perform Principal Component Analysis"""
        from sklearn.decomposition import PCA
        
        # Standardize data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        # Perform PCA
        pca = PCA()
        pca_result = pca.fit_transform(data_scaled)
        
        # Explained variance
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        # Component loadings
        loadings = pd.DataFrame(
            pca.components_.T,
            columns=[f'PC{i+1}' for i in range(pca.n_components_)],
            index=data.columns
        )
        
        return {
            'explained_variance': explained_variance,
            'cumulative_variance': cumulative_variance,
            'loadings': loadings,
            'n_components': pca.n_components_,
            'components_to_explain_80_percent': np.argmax(cumulative_variance >= 0.8) + 1
        }
    
    def perform_granger_causality(self, target: str, predictor: str, 
                                max_lags: int = 4) -> Dict:
        """
        Perform Granger causality test
        
        Args:
            target: Target variable
            predictor: Predictor variable
            max_lags: Maximum number of lags to test
            
        Returns:
            Dictionary with Granger causality test results
        """
        try:
            from statsmodels.tsa.stattools import grangercausalitytests
            
            # Prepare data
            growth_data = self.data[[target, predictor]].pct_change().dropna()
            
            # Perform Granger causality test
            test_data = growth_data[[predictor, target]]  # Note: order matters
            gc_result = grangercausalitytests(test_data, maxlag=max_lags, verbose=False)
            
            # Extract results
            results = {}
            for lag in range(1, max_lags + 1):
                if lag in gc_result:
                    lag_result = gc_result[lag]
                    results[lag] = {
                        'f_statistic': lag_result[0]['ssr_ftest'][0],
                        'p_value': lag_result[0]['ssr_ftest'][1],
                        'is_significant': lag_result[0]['ssr_ftest'][1] < 0.05
                    }
            
            # Overall result (use minimum p-value)
            min_p_value = min([result['p_value'] for result in results.values()])
            overall_significant = min_p_value < 0.05
            
            return {
                'results_by_lag': results,
                'min_p_value': min_p_value,
                'is_causal': overall_significant,
                'optimal_lag': min(results.keys(), key=lambda k: results[k]['p_value'])
            }
            
        except Exception as e:
            logger.error(f"Granger causality test failed: {e}")
            return {'error': str(e)}
    
    def generate_statistical_report(self, regression_results: Dict = None,
                                  correlation_results: Dict = None,
                                  causality_results: Dict = None) -> str:
        """
        Generate comprehensive statistical analysis report
        
        Args:
            regression_results: Results from regression analysis
            correlation_results: Results from correlation analysis
            causality_results: Results from causality analysis
            
        Returns:
            Formatted report string
        """
        report = "STATISTICAL MODELING REPORT\n"
        report += "=" * 50 + "\n\n"
        
        if regression_results:
            report += "REGRESSION ANALYSIS\n"
            report += "-" * 30 + "\n"
            
            # Model performance
            performance = regression_results['performance']
            report += f"Model Performance:\n"
            report += f"  R²: {performance['r2']:.4f}\n"
            report += f"  RMSE: {performance['rmse']:.4f}\n"
            report += f"  MAE: {performance['mae']:.4f}\n\n"
            
            # Top coefficients
            coefficients = regression_results['coefficients']
            report += f"Top 5 Most Important Variables:\n"
            for i, row in coefficients.head().iterrows():
                report += f"  {row['variable']}: {row['coefficient']:.4f}\n"
            report += "\n"
            
            # Diagnostics
            diagnostics = regression_results['diagnostics']
            report += f"Model Diagnostics:\n"
            
            if 'normality' in diagnostics and 'error' not in diagnostics['normality']:
                norm = diagnostics['normality']
                report += f"  Normality (Shapiro-Wilk): p={norm['p_value']:.4f} "
                report += f"({'Normal' if norm['is_normal'] else 'Not Normal'})\n"
            
            if 'homoscedasticity' in diagnostics and 'error' not in diagnostics['homoscedasticity']:
                hom = diagnostics['homoscedasticity']
                report += f"  Homoscedasticity (Breusch-Pagan): p={hom['p_value']:.4f} "
                report += f"({'Homoscedastic' if hom['is_homoscedastic'] else 'Heteroscedastic'})\n"
            
            if 'autocorrelation' in diagnostics and 'error' not in diagnostics['autocorrelation']:
                autocorr = diagnostics['autocorrelation']
                report += f"  Autocorrelation (Durbin-Watson): {autocorr['statistic']:.4f} "
                report += f"({autocorr['interpretation']})\n"
            
            if 'multicollinearity' in diagnostics and 'error' not in diagnostics['multicollinearity']:
                mult = diagnostics['multicollinearity']
                report += f"  Multicollinearity (VIF): Mean VIF = {mult['mean_vif']:.2f}\n"
                if mult['high_vif_variables']:
                    report += f"    High VIF variables: {', '.join(mult['high_vif_variables'])}\n"
            
            report += "\n"
        
        if correlation_results:
            report += "CORRELATION ANALYSIS\n"
            report += "-" * 30 + "\n"
            report += f"Method: {correlation_results['method'].title()}\n"
            report += f"Significant Correlations: {len(correlation_results['significant_correlations'])}\n\n"
            
            # Top correlations
            report += f"Top 5 Strongest Correlations:\n"
            for i, corr in enumerate(correlation_results['significant_correlations'][:5]):
                report += f"  {corr['variable1']} ↔ {corr['variable2']}: "
                report += f"{corr['correlation']:.4f} ({corr['strength']}, p={corr['p_value']:.4f})\n"
            
            # PCA results
            if 'pca_analysis' in correlation_results and 'error' not in correlation_results['pca_analysis']:
                pca = correlation_results['pca_analysis']
                report += f"\nPrincipal Component Analysis:\n"
                report += f"  Components to explain 80% variance: {pca['components_to_explain_80_percent']}\n"
                report += f"  Total components: {pca['n_components']}\n"
            
            report += "\n"
        
        if causality_results:
            report += "GRANGER CAUSALITY ANALYSIS\n"
            report += "-" * 30 + "\n"
            
            for target, results in causality_results.items():
                if 'error' not in results:
                    report += f"{target}:\n"
                    report += f"  Is causal: {results['is_causal']}\n"
                    report += f"  Minimum p-value: {results['min_p_value']:.4f}\n"
                    report += f"  Optimal lag: {results['optimal_lag']}\n\n"
        
        return report 