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
        
    def prepare_data(self, target_series: str, frequency: str = 'Q') -> pd.Series:
        """
        Prepare time series data for forecasting
        
        Args:
            target_series: Series name to forecast
            frequency: Data frequency ('Q' for quarterly, 'M' for monthly)
            
        Returns:
            Prepared time series
        """
        if target_series not in self.data.columns:
            raise ValueError(f"Series {target_series} not found in data")
            
        series = self.data[target_series].dropna()
        
        # Resample to desired frequency
        if frequency == 'Q':
            series = series.resample('Q').mean()
        elif frequency == 'M':
            series = series.resample('M').mean()
            
        # Calculate growth rates for economic indicators
        if target_series in ['GDPC1', 'INDPRO', 'RSAFS']:
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
        Fit ARIMA model to time series
        
        Args:
            series: Time series data
            order: ARIMA order (p, d, q). If None, auto-detect
            
        Returns:
            Fitted ARIMA model
        """
        if order is None:
            # Auto-detect order using AIC minimization
            best_aic = np.inf
            best_order = (1, 1, 1)
            
            for p in range(0, 3):
                for d in range(0, 2):
                    for q in range(0, 3):
                        try:
                            model = ARIMA(series, order=(p, d, q))
                            fitted_model = model.fit()
                            if fitted_model.aic < best_aic:
                                best_aic = fitted_model.aic
                                best_order = (p, d, q)
                        except:
                            continue
            
            order = best_order
            logger.info(f"Auto-detected ARIMA order: {order}")
        
        model = ARIMA(series, order=order)
        fitted_model = model.fit()
        
        return fitted_model
    
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
        
        # Generate forecast
        forecast = model.forecast(steps=forecast_periods)
        
        # Calculate confidence intervals
        if model_type == 'arima':
            forecast_ci = model.get_forecast(steps=forecast_periods).conf_int()
        else:
            # For ETS, use simple confidence intervals
            forecast_std = series.std()
            forecast_ci = pd.DataFrame({
                'lower': forecast - 1.96 * forecast_std,
                'upper': forecast + 1.96 * forecast_std
            })
        
        return {
            'model': model,
            'model_type': model_type,
            'forecast': forecast,
            'confidence_intervals': forecast_ci,
            'aic': model.aic if hasattr(model, 'aic') else None
        }
    
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
        mape = np.mean(np.abs(np.array(actual_values) - np.array(predicted_values)) / np.abs(actual_values)) * 100
        
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
                # Prepare data
                series = self.prepare_data(indicator)
                
                # Check stationarity
                stationarity = self.check_stationarity(series)
                
                # Decompose series
                decomposition = self.decompose_series(series)
                
                # Generate forecast
                forecast_result = self.forecast_series(series)
                
                # Perform backtest
                backtest_result = self.backtest_forecast(series)
                
                results[indicator] = {
                    'stationarity': stationarity,
                    'decomposition': decomposition,
                    'forecast': forecast_result,
                    'backtest': backtest_result,
                    'series': series
                }
                
                logger.info(f"Successfully forecasted {indicator}")
                
            except Exception as e:
                logger.error(f"Failed to forecast {indicator}: {e}")
                results[indicator] = {'error': str(e)}
        
        return results
    
    def generate_forecast_report(self, forecasts: Dict) -> str:
        """
        Generate comprehensive forecast report
        
        Args:
            forecasts: Dictionary with forecast results
            
        Returns:
            Formatted report string
        """
        report = "ECONOMIC FORECASTING REPORT\n"
        report += "=" * 50 + "\n\n"
        
        for indicator, result in forecasts.items():
            if 'error' in result:
                report += f"{indicator}: ERROR - {result['error']}\n\n"
                continue
                
            report += f"INDICATOR: {indicator}\n"
            report += "-" * 30 + "\n"
            
            # Stationarity results
            stationarity = result['stationarity']
            report += f"Stationarity Test (ADF):\n"
            report += f"  ADF Statistic: {stationarity['adf_statistic']:.4f}\n"
            report += f"  P-value: {stationarity['p_value']:.4f}\n"
            report += f"  Is Stationary: {stationarity['is_stationary']}\n\n"
            
            # Model information
            forecast = result['forecast']
            report += f"Model: {forecast['model_type'].upper()}\n"
            if forecast['aic']:
                report += f"AIC: {forecast['aic']:.4f}\n"
            report += f"Forecast Periods: {len(forecast['forecast'])}\n\n"
            
            # Backtest results
            backtest = result['backtest']
            if 'error' not in backtest:
                report += f"Backtest Performance:\n"
                report += f"  MAE: {backtest['mae']:.4f}\n"
                report += f"  RMSE: {backtest['rmse']:.4f}\n"
                report += f"  MAPE: {backtest['mape']:.2f}%\n"
                report += f"  Test Periods: {backtest['test_periods']}\n\n"
            
            # Forecast values
            report += f"Forecast Values:\n"
            for i, value in enumerate(forecast['forecast']):
                ci = forecast['confidence_intervals']
                lower = ci.iloc[i]['lower'] if 'lower' in ci.columns else 'N/A'
                upper = ci.iloc[i]['upper'] if 'upper' in ci.columns else 'N/A'
                report += f"  Period {i+1}: {value:.4f} [{lower:.4f}, {upper:.4f}]\n"
            
            report += "\n" + "=" * 50 + "\n\n"
        
        return report 