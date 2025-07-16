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
        try:
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
        except Exception as e:
            return {'error': f'Regression model fitting failed: {str(e)}'}
    
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
            shapiro_stat, shapiro_p = stats.shapiro(residuals)
            diagnostics['normality'] = {
                'test': 'Shapiro-Wilk',
                'statistic': shapiro_stat,
                'p_value': shapiro_p,
                'interpretation': self._interpret_normality(shapiro_p)
            }
        except Exception as e:
            diagnostics['normality'] = {'error': str(e)}
        
        # 2. Homoscedasticity test (Breusch-Pagan)
        try:
            bp_stat, bp_p, bp_f, bp_f_p = het_breuschpagan(residuals, features)
            diagnostics['homoscedasticity'] = {
                'test': 'Breusch-Pagan',
                'statistic': bp_stat,
                'p_value': bp_p,
                'interpretation': self._interpret_homoscedasticity(bp_p)
            }
        except Exception as e:
            diagnostics['homoscedasticity'] = {'error': str(e)}
        
        # 3. Autocorrelation test (Durbin-Watson)
        try:
            dw_stat = durbin_watson(residuals)
            diagnostics['autocorrelation'] = {
                'test': 'Durbin-Watson',
                'statistic': dw_stat,
                'interpretation': self._interpret_durbin_watson(dw_stat)
            }
        except Exception as e:
            diagnostics['autocorrelation'] = {'error': str(e)}
        
        # 4. Multicollinearity (VIF)
        try:
            vif_data = []
            for i in range(features.shape[1]):
                vif = variance_inflation_factor(features.values, i)
                vif_data.append({
                    'variable': features.columns[i],
                    'vif': vif
                })
            diagnostics['multicollinearity'] = {
                'test': 'Variance Inflation Factor',
                'vif_values': vif_data,
                'interpretation': self._interpret_multicollinearity(vif_data)
            }
        except Exception as e:
            diagnostics['multicollinearity'] = {'error': str(e)}
        
        return diagnostics
    
    def _interpret_normality(self, p_value: float) -> str:
        """Interpret normality test results"""
        if p_value < 0.05:
            return "Residuals are not normally distributed (p < 0.05)"
        else:
            return "Residuals appear to be normally distributed (p >= 0.05)"
    
    def _interpret_homoscedasticity(self, p_value: float) -> str:
        """Interpret homoscedasticity test results"""
        if p_value < 0.05:
            return "Heteroscedasticity detected (p < 0.05)"
        else:
            return "Homoscedasticity assumption appears valid (p >= 0.05)"
    
    def _interpret_durbin_watson(self, dw_stat: float) -> str:
        """Interpret Durbin-Watson test results"""
        if dw_stat < 1.5:
            return "Positive autocorrelation detected"
        elif dw_stat > 2.5:
            return "Negative autocorrelation detected"
        else:
            return "No significant autocorrelation"
    
    def _interpret_multicollinearity(self, vif_data: List[Dict]) -> str:
        """Interpret multicollinearity test results"""
        high_vif = [item for item in vif_data if item['vif'] > 10]
        if high_vif:
            return f"Multicollinearity detected in {len(high_vif)} variables"
        else:
            return "No significant multicollinearity detected"
    
    def analyze_correlations(self, indicators: List[str] = None, 
                           method: str = 'pearson') -> Dict:
        """
        Analyze correlations between economic indicators
        
        Args:
            indicators: List of indicators to analyze. If None, use all numeric columns
            method: Correlation method ('pearson', 'spearman', 'kendall')
            
        Returns:
            Dictionary with correlation analysis results
        """
        if indicators is None:
            indicators = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Calculate correlation matrix
        corr_matrix = self.data[indicators].corr(method=method)
        
        # Find strongest correlations
        corr_pairs = []
        for i in range(len(indicators)):
            for j in range(i+1, len(indicators)):
                corr_value = corr_matrix.iloc[i, j]
                corr_pairs.append({
                    'variable1': indicators[i],
                    'variable2': indicators[j],
                    'correlation': corr_value,
                    'strength': self._interpret_correlation_strength(corr_value)
                })
        
        # Sort by absolute correlation value
        corr_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        return {
            'correlation_matrix': corr_matrix,
            'correlation_pairs': corr_pairs,
            'method': method,
            'strongest_correlations': corr_pairs[:5]
        }
    
    def _interpret_correlation_strength(self, corr_value: float) -> str:
        """Interpret correlation strength"""
        abs_corr = abs(corr_value)
        if abs_corr >= 0.8:
            return "Very strong"
        elif abs_corr >= 0.6:
            return "Strong"
        elif abs_corr >= 0.4:
            return "Moderate"
        elif abs_corr >= 0.2:
            return "Weak"
        else:
            return "Very weak"
    
    def perform_stationarity_tests(self, series: pd.Series) -> Dict:
        """
        Perform stationarity tests on time series data
        
        Args:
            series: Time series data
            
        Returns:
            Dictionary with stationarity test results
        """
        results = {}
        
        # ADF test
        try:
            adf_stat, adf_p, adf_critical = adfuller(series.dropna())
            results['adf'] = {
                'statistic': adf_stat,
                'p_value': adf_p,
                'critical_values': adf_critical,
                'is_stationary': adf_p < 0.05
            }
        except Exception as e:
            results['adf'] = {'error': str(e)}
        
        # KPSS test
        try:
            kpss_stat, kpss_p, kpss_critical = kpss(series.dropna())
            results['kpss'] = {
                'statistic': kpss_stat,
                'p_value': kpss_p,
                'critical_values': kpss_critical,
                'is_stationary': kpss_p >= 0.05
            }
        except Exception as e:
            results['kpss'] = {'error': str(e)}
        
        return results
    
    def _perform_pca_analysis(self, data: pd.DataFrame) -> Dict:
        """
        Perform Principal Component Analysis
        
        Args:
            data: Standardized data matrix
            
        Returns:
            Dictionary with PCA results
        """
        from sklearn.decomposition import PCA
        
        pca = PCA()
        pca.fit(data)
        
        # Explained variance
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        return {
            'components': pca.components_,
            'explained_variance': explained_variance,
            'cumulative_variance': cumulative_variance,
            'n_components': len(explained_variance)
        }
    
    def perform_granger_causality(self, target: str, predictor: str, 
                                max_lags: int = 4) -> Dict:
        """
        Perform Granger causality test
        
        Args:
            target: Target variable name
            predictor: Predictor variable name
            max_lags: Maximum number of lags to test
            
        Returns:
            Dictionary with Granger causality test results
        """
        try:
            from statsmodels.tsa.stattools import grangercausalitytests
            
            # Prepare data
            data = self.data[[target, predictor]].dropna()
            
            if len(data) < max_lags + 10:
                return {'error': 'Insufficient data for Granger causality test'}
            
            # Perform test
            gc_result = grangercausalitytests(data, maxlag=max_lags, verbose=False)
            
            # Extract results
            results = {}
            for lag in range(1, max_lags + 1):
                if lag in gc_result:
                    f_stat = gc_result[lag][0]['ssr_ftest']
                    results[f'lag_{lag}'] = {
                        'f_statistic': f_stat[0],
                        'p_value': f_stat[1],
                        'significant': f_stat[1] < 0.05
                    }
            
            return {
                'target': target,
                'predictor': predictor,
                'max_lags': max_lags,
                'results': results
            }
        except Exception as e:
            return {'error': f'Granger causality test failed: {str(e)}'}
    
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
        report = []
        report.append("=== STATISTICAL ANALYSIS REPORT ===\n")
        
        # Regression results
        if regression_results and 'error' not in regression_results:
            report.append("REGRESSION ANALYSIS:")
            perf = regression_results['performance']
            report.append(f"- R² Score: {perf['r2']:.4f}")
            report.append(f"- RMSE: {perf['rmse']:.4f}")
            report.append(f"- MAE: {perf['mae']:.4f}")
            
            # Top coefficients
            top_coeffs = regression_results['coefficients'].head(5)
            report.append("- Top 5 coefficients:")
            for _, row in top_coeffs.iterrows():
                report.append(f"  {row['variable']}: {row['coefficient']:.4f}")
            report.append("")
        
        # Correlation results
        if correlation_results:
            report.append("CORRELATION ANALYSIS:")
            strongest = correlation_results.get('strongest_correlations', [])
            for pair in strongest[:3]:
                report.append(f"- {pair['variable1']} ↔ {pair['variable2']}: "
                           f"{pair['correlation']:.3f} ({pair['strength']})")
            report.append("")
        
        # Causality results
        if causality_results and 'error' not in causality_results:
            report.append("GRANGER CAUSALITY ANALYSIS:")
            results = causality_results.get('results', {})
            significant_lags = [lag for lag, result in results.items() 
                              if result.get('significant', False)]
            if significant_lags:
                report.append(f"- Significant causality detected at lags: {', '.join(significant_lags)}")
            else:
                report.append("- No significant causality detected")
            report.append("")
        
        return "\n".join(report) 