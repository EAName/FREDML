"""
Economic Forecasting Module
Advanced time series forecasting for economic indicators using ARIMA/ETS models
"""

import logging
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

logger = logging.getLogger(__name__)

class EconomicForecaster:
    """
    Advanced economic forecasting using ARIMA and ETS models
    with comprehensive backtesting and performance evaluation
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize forecaster with economic data
        
        Args:
            data: DataFrame with economic indicators (GDPC1, INDPRO, RSAFS, etc.)
        """
        self.data = data.copy()
        self.forecasts = {}
        self.backtest_results = {}
        self.model_performance = {}
        
    def prepare_data(self, target_series: str, frequency: str = 'Q', for_arima: bool = True) -> pd.Series:
        """
        Prepare time series data for forecasting or analysis.
        Args:
            target_series: Series name to forecast
            frequency: Data frequency ('Q' for quarterly, 'M' for monthly)
            for_arima: If True, returns raw levels for ARIMA; if False, returns growth rate
        Returns:
            Prepared time series
        """
        if target_series not in self.data.columns:
            raise ValueError(f"Series {target_series} not found in data")
        series = self.data[target_series].dropna()
        # Ensure time-based index
        if not isinstance(series.index, pd.DatetimeIndex):
            raise ValueError("Index must be datetime type")
        # Resample to desired frequency
        if frequency == 'Q':
            series = series.resample('Q').mean()
        elif frequency == 'M':
            series = series.resample('M').mean()
        # Only use growth rates if for_arima is False
        if not for_arima and target_series in ['GDPC1', 'INDPRO', 'RSAFS']:
            series = series.pct_change().dropna()
        return series
    
    def check_stationarity(self, series: pd.Series) -> Dict:
        """
        Perform Augmented Dickey-Fuller test for stationarity
        
        Args:
            series: Time series to test
            
        Returns:
            Dictionary with test results
        """
        result = adfuller(series.dropna())
        
        return {
            'adf_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[4],
            'is_stationary': result[1] < 0.05
        }
    
    def decompose_series(self, series: pd.Series, period: int = 4) -> Dict:
        """
        Decompose time series into trend, seasonal, and residual components
        
        Args:
            series: Time series to decompose
            period: Seasonal period (4 for quarterly, 12 for monthly)
            
        Returns:
            Dictionary with decomposition components
        """
        decomposition = seasonal_decompose(series.dropna(), period=period, extrapolate_trend='freq')
        
        return {
            'trend': decomposition.trend,
            'seasonal': decomposition.seasonal,
            'residual': decomposition.resid,
            'observed': decomposition.observed
        }
    
    def fit_arima_model(self, series: pd.Series, order: Tuple[int, int, int] = None) -> ARIMA:
        """
        Fit ARIMA model to time series using raw levels (not growth rates)
        
        Args:
            series: Time series data (raw levels)
            order: ARIMA order (p, d, q). If None, auto-detect
            
        Returns:
            Fitted ARIMA model
        """
        # Ensure we're working with raw levels, not growth rates
        if series.isna().any():
            series = series.dropna()
        
        # Ensure series has enough data points
        if len(series) < 10:
            raise ValueError("Series must have at least 10 data points for ARIMA fitting")
        

        
        if order is None:
            # Auto-detect order using AIC minimization with improved search
            best_aic = np.inf
            best_order = (1, 1, 1)
            
            # Improved order search that avoids degenerate models
            # Start with more reasonable orders to avoid ARIMA(0,0,0)
            search_orders = [
                (1, 1, 1), (2, 1, 1), (1, 1, 2), (2, 1, 2),  # Common orders
                (0, 1, 1), (1, 0, 1), (1, 1, 0),  # Simple orders
                (2, 0, 1), (1, 0, 2), (2, 1, 0),  # Alternative orders
                (3, 1, 1), (1, 1, 3), (2, 2, 1), (1, 2, 2),  # Higher orders
            ]
            
            for p, d, q in search_orders:
                try:
                    model = ARIMA(series, order=(p, d, q))
                    fitted_model = model.fit()
                    
                    # Check if model is degenerate (all parameters near zero)
                    params = fitted_model.params
                    if len(params) > 0:
                        # Skip models where all AR/MA parameters are very small
                        ar_params = params[1:p+1] if p > 0 else []
                        ma_params = params[p+1:p+1+q] if q > 0 else []
                        
                        # Check if model is essentially a random walk or constant
                        if (p == 0 and d == 0 and q == 0) or \
                           (p == 0 and d == 1 and q == 0) or \
                           (len(ar_params) > 0 and all(abs(p) < 0.01 for p in ar_params)) or \
                           (len(ma_params) > 0 and all(abs(p) < 0.01 for p in ma_params)):
                            logger.debug(f"Skipping degenerate ARIMA({p},{d},{q})")
                            continue
                    
                    if fitted_model.aic < best_aic:
                        best_aic = fitted_model.aic
                        best_order = (p, d, q)
                        logger.debug(f"New best ARIMA({p},{d},{q}) with AIC: {best_aic}")
                        
                except Exception as e:
                    logger.debug(f"ARIMA({p},{d},{q}) failed: {e}")
                    continue
            
            order = best_order
            logger.info(f"Auto-detected ARIMA order: {order} with AIC: {best_aic}")
            
            # If we still have a degenerate model, force a reasonable order
            if order == (0, 0, 0) or order == (0, 1, 0):
                logger.warning("Detected degenerate ARIMA order, forcing to ARIMA(1,1,1)")
                order = (1, 1, 1)
        
        try:
            model = ARIMA(series, order=order)
            fitted_model = model.fit()
            
            # Debug: Log model parameters
            logger.info(f"ARIMA model fitted successfully with AIC: {fitted_model.aic}")
            logger.info(f"ARIMA order: {order}")
            logger.info(f"Model parameters: {fitted_model.params}")
            
            return fitted_model
        except Exception as e:
            logger.warning(f"ARIMA fitting failed with order {order}: {e}")
            # Try fallback orders
            fallback_orders = [(1, 1, 1), (0, 1, 1), (1, 0, 1), (1, 1, 0)]
            for fallback_order in fallback_orders:
                try:
                    model = ARIMA(series, order=fallback_order)
                    fitted_model = model.fit()
                    logger.info(f"ARIMA fallback model fitted with order {fallback_order}")
                    return fitted_model
                except Exception as fallback_e:
                    logger.debug(f"Fallback ARIMA{fallback_order} failed: {fallback_e}")
                    continue
            
            # Last resort: simple moving average
            logger.warning("All ARIMA models failed, using simple moving average")
            raise ValueError("Unable to fit any ARIMA model to the data")
    
    def fit_ets_model(self, series: pd.Series, seasonal_periods: int = 4) -> ExponentialSmoothing:
        """
        Fit ETS (Exponential Smoothing) model to time series
        
        Args:
            series: Time series data
            seasonal_periods: Number of seasonal periods
            
        Returns:
            Fitted ETS model
        """
        model = ExponentialSmoothing(
            series,
            seasonal_periods=seasonal_periods,
            trend='add',
            seasonal='add'
        )
        fitted_model = model.fit()
        
        return fitted_model
    
    def forecast_series(self, series: pd.Series, model_type: str = 'auto', 
                       forecast_periods: int = 4) -> Dict:
        """
        Forecast time series using specified model
        
        Args:
            series: Time series to forecast
            model_type: 'arima', 'ets', or 'auto'
            forecast_periods: Number of periods to forecast
            
        Returns:
            Dictionary with forecast results
        """
        if model_type == 'auto':
            # Try both models and select the one with better AIC
            try:
                arima_model = self.fit_arima_model(series)
                arima_aic = arima_model.aic
            except:
                arima_aic = np.inf
                
            try:
                ets_model = self.fit_ets_model(series)
                ets_aic = ets_model.aic
            except:
                ets_aic = np.inf
            
            if arima_aic < ets_aic:
                model_type = 'arima'
                model = arima_model
            else:
                model_type = 'ets'
                model = ets_model
        elif model_type == 'arima':
            model = self.fit_arima_model(series)
        elif model_type == 'ets':
            model = self.fit_ets_model(series)
        else:
            raise ValueError("model_type must be 'arima', 'ets', or 'auto'")
        
        # Generate forecast using proper method for each model type
        if model_type == 'arima':
            # Use get_forecast() for ARIMA to get proper confidence intervals
            forecast_result = model.get_forecast(steps=forecast_periods)
            forecast = forecast_result.predicted_mean
            

            
            try:
                forecast_ci = forecast_result.conf_int()
                # Check if confidence intervals are valid (not all NaN)
                if forecast_ci.isna().all().all() or forecast_ci.empty:
                    # Improved fallback confidence intervals
                    forecast_ci = self._calculate_improved_confidence_intervals(forecast, series, model)
                else:
                    # Ensure confidence intervals have proper column names
                    if len(forecast_ci.columns) >= 2:
                        forecast_ci.columns = ['lower', 'upper']
                    else:
                        # Improved fallback if column structure is unexpected
                        forecast_ci = self._calculate_improved_confidence_intervals(forecast, series, model)
                
                # Debug: Log confidence intervals
                logger.info(f"ARIMA confidence intervals: {forecast_ci.to_dict()}")
                
                # Check if confidence intervals are too wide and provide warning
                ci_widths = forecast_ci['upper'] - forecast_ci['lower']
                mean_width = ci_widths.mean()
                forecast_mean = forecast.mean()
                relative_width = mean_width / abs(forecast_mean) if abs(forecast_mean) > 0 else 0
                
                if relative_width > 0.5:  # If confidence interval is more than 50% of forecast value
                    logger.warning(f"Confidence intervals are very wide (relative width: {relative_width:.2%})")
                    logger.info("This may indicate high uncertainty or model instability")
                
            except Exception as e:
                logger.warning(f"ARIMA confidence interval calculation failed: {e}")
                # Improved fallback confidence intervals
                forecast_ci = self._calculate_improved_confidence_intervals(forecast, series, model)
        else:
            # For ETS, use forecast() method
            forecast = model.forecast(steps=forecast_periods)
            # Use improved confidence intervals for ETS
            forecast_ci = self._calculate_improved_confidence_intervals(forecast, series, model)
        
        # Debug: Log final results
        logger.info(f"Final forecast is flat: {len(set(forecast)) == 1}")
        logger.info(f"Forecast type: {type(forecast)}")
        
        return {
            'model': model,
            'model_type': model_type,
            'forecast': forecast,
            'confidence_intervals': forecast_ci,
            'aic': model.aic if hasattr(model, 'aic') else None
        }
    
    def _calculate_improved_confidence_intervals(self, forecast: pd.Series, series: pd.Series, model) -> pd.DataFrame:
        """
        Calculate improved confidence intervals with better uncertainty quantification
        
        Args:
            forecast: Forecast values
            series: Original time series
            model: Fitted model
            
        Returns:
            DataFrame with improved confidence intervals
        """
        try:
            # Calculate forecast errors from model residuals if available
            if hasattr(model, 'resid') and len(model.resid) > 0:
                # Use model residuals for more accurate uncertainty
                residuals = model.resid.dropna()
                forecast_std = residuals.std()
                
                # Adjust for forecast horizon (uncertainty increases with horizon)
                horizon_factors = np.sqrt(np.arange(1, len(forecast) + 1))
                confidence_intervals = []
                
                for i, (fcast, factor) in enumerate(zip(forecast, horizon_factors)):
                    # Use 95% confidence interval (1.96 * std)
                    margin = 1.96 * forecast_std * factor
                    lower = fcast - margin
                    upper = fcast + margin
                    confidence_intervals.append({'lower': lower, 'upper': upper})
                
                return pd.DataFrame(confidence_intervals, index=forecast.index)
            
            else:
                # Fallback to series-based uncertainty
                series_std = series.std()
                # Use a more conservative approach for economic data
                # Economic forecasts typically have higher uncertainty
                uncertainty_factor = 1.5  # Adjust based on data characteristics
                
                confidence_intervals = []
                for i, fcast in enumerate(forecast):
                    # Increase uncertainty with forecast horizon
                    horizon_factor = 1 + (i * 0.1)  # 10% increase per period
                    margin = 1.96 * series_std * uncertainty_factor * horizon_factor
                    lower = fcast - margin
                    upper = fcast + margin
                    confidence_intervals.append({'lower': lower, 'upper': upper})
                
                return pd.DataFrame(confidence_intervals, index=forecast.index)
                
        except Exception as e:
            logger.warning(f"Improved confidence interval calculation failed: {e}")
            # Ultimate fallback
            series_std = series.std()
            return pd.DataFrame({
                'lower': forecast - 1.96 * series_std,
                'upper': forecast + 1.96 * series_std
            }, index=forecast.index)
    
    def backtest_forecast(self, series: pd.Series, model_type: str = 'auto',
                         train_size: float = 0.8, test_periods: int = 8) -> Dict:
        """
        Perform backtesting of forecasting models
        
        Args:
            series: Time series to backtest
            model_type: Model type to use
            train_size: Proportion of data for training
            test_periods: Number of periods to test
            
        Returns:
            Dictionary with backtest results
        """
        n = len(series)
        train_end = int(n * train_size)
        
        actual_values = []
        predicted_values = []
        errors = []
        
        for i in range(test_periods):
            if train_end + i >= n:
                break
                
            # Use expanding window
            train_data = series.iloc[:train_end + i]
            test_value = series.iloc[train_end + i]
            
            try:
                forecast_result = self.forecast_series(train_data, model_type, 1)
                prediction = forecast_result['forecast'].iloc[0]
                
                actual_values.append(test_value)
                predicted_values.append(prediction)
                errors.append(test_value - prediction)
                
            except Exception as e:
                logger.warning(f"Forecast failed at step {i}: {e}")
                continue
        
        if not actual_values:
            return {'error': 'No successful forecasts generated'}
        
        # Calculate performance metrics
        mae = mean_absolute_error(actual_values, predicted_values)
        mse = mean_squared_error(actual_values, predicted_values)
        rmse = np.sqrt(mse)
        
        # Use safe MAPE calculation to avoid division by zero
        actual_array = np.array(actual_values)
        predicted_array = np.array(predicted_values)
        denominator = np.maximum(np.abs(actual_array), 1e-8)
        mape = np.mean(np.abs((actual_array - predicted_array) / denominator)) * 100
        
        return {
            'actual_values': actual_values,
            'predicted_values': predicted_values,
            'errors': errors,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape,
            'test_periods': len(actual_values)
        }
    
    def forecast_economic_indicators(self, indicators: List[str] = None) -> Dict:
        """
        Forecast multiple economic indicators
        
        Args:
            indicators: List of indicators to forecast. If None, use default set
            
        Returns:
            Dictionary with forecasts for all indicators
        """
        if indicators is None:
            indicators = ['GDPC1', 'INDPRO', 'RSAFS']
        
        results = {}
        
        for indicator in indicators:
            try:
                # Prepare raw data for forecasting (use raw levels, not growth rates)
                series = self.prepare_data(indicator, for_arima=True)
                
                # Prepare growth rates for analysis
                growth_series = self.prepare_data(indicator, for_arima=False)
                
                # Check stationarity on growth rates
                stationarity = self.check_stationarity(growth_series)
                
                # Decompose growth rates
                decomposition = self.decompose_series(growth_series)
                
                # Generate forecast using raw levels
                forecast_result = self.forecast_series(series)
                
                # Perform backtest on raw levels
                backtest_result = self.backtest_forecast(series)
                
                results[indicator] = {
                    'stationarity': stationarity,
                    'decomposition': decomposition,
                    'forecast': forecast_result,
                    'backtest': backtest_result,
                    'raw_series': series,
                    'growth_series': growth_series
                }
                
                logger.info(f"Successfully forecasted {indicator}")
                
            except Exception as e:
                logger.error(f"Failed to forecast {indicator}: {e}")
                results[indicator] = {'error': str(e)}
        
        return results
    
    def generate_forecast_report(self, forecast_result, periods=None):
        """
        Generate a markdown table for forecast results.
        Args:
            forecast_result: dict with keys 'forecast', 'confidence_intervals'
            periods: list of period labels (optional)
        Returns:
            Markdown string
        """
        forecast = forecast_result.get('forecast')
        ci = forecast_result.get('confidence_intervals')
        if forecast is None or ci is None:
            return 'No forecast results available.'
        if periods is None:
            periods = [f"Period {i+1}" for i in range(len(forecast))]
        lines = ["| Period  | Forecast      | 95% CI Lower | 95% CI Upper |", "| ------- | ------------- | ------------ | ------------ |"]
        for i, (f, p) in enumerate(zip(forecast, periods)):
            try:
                lower = ci.iloc[i, 0] if hasattr(ci, 'iloc') else ci[i][0]
                upper = ci.iloc[i, 1] if hasattr(ci, 'iloc') else ci[i][1]
            except Exception:
                lower = upper = 'N/A'
            lines.append(f"| {p} | **{f:,.2f}** | {lower if isinstance(lower, str) else f'{lower:,.2f}'} | {upper if isinstance(upper, str) else f'{upper:,.2f}'} |")
        return '\n'.join(lines) 