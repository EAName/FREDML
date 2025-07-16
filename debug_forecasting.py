#!/usr/bin/env python3
"""
Debug script to test forecasting and identify why forecasts are flat
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from core.fred_client import FREDDataCollectorV2
from analysis.economic_forecasting import EconomicForecaster
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_forecasting():
    """Debug the forecasting process"""
    
    # Initialize FRED data collector
    api_key = os.getenv('FRED_API_KEY')
    if not api_key:
        logger.error("FRED_API_KEY not found in environment")
        return
    
    collector = FREDDataCollectorV2(api_key)
    
    # Fetch data
    indicators = ['GDPC1', 'INDPRO', 'RSAFS']
    data_dict = collector.get_economic_data(indicators, start_date='2020-01-01', end_date='2024-12-31')
    df = collector.create_dataframe(data_dict)
    
    if df.empty:
        logger.error("No data fetched")
        return
    
    logger.info(f"Fetched data shape: {df.shape}")
    logger.info(f"Data columns: {df.columns.tolist()}")
    logger.info(f"Data index: {df.index[:5]} to {df.index[-5:]}")
    
    # Initialize forecaster
    forecaster = EconomicForecaster(df)
    
    # Test each indicator
    for indicator in indicators:
        logger.info(f"\n{'='*50}")
        logger.info(f"Testing {indicator}")
        logger.info(f"{'='*50}")
        
        # Get raw data
        raw_series = forecaster.prepare_data(indicator, for_arima=True)
        growth_series = forecaster.prepare_data(indicator, for_arima=False)
        
        logger.info(f"Raw series shape: {raw_series.shape}")
        logger.info(f"Raw series head: {raw_series.head()}")
        logger.info(f"Raw series tail: {raw_series.tail()}")
        logger.info(f"Raw series stats: mean={raw_series.mean():.2f}, std={raw_series.std():.2f}")
        logger.info(f"Raw series range: {raw_series.min():.2f} to {raw_series.max():.2f}")
        
        logger.info(f"Growth series shape: {growth_series.shape}")
        logger.info(f"Growth series head: {growth_series.head()}")
        logger.info(f"Growth series stats: mean={growth_series.mean():.4f}, std={growth_series.std():.4f}")
        
        # Test ARIMA fitting
        try:
            model = forecaster.fit_arima_model(raw_series)
            logger.info(f"ARIMA model fitted successfully: {model}")
            # Fix the order access
            try:
                order = model.model.order
            except:
                try:
                    order = model.model_orders
                except:
                    order = "Unknown"
            logger.info(f"ARIMA order: {order}")
            logger.info(f"ARIMA AIC: {model.aic}")
            
            # Test forecasting
            forecast_result = forecaster.forecast_series(raw_series, model_type='arima')
            forecast = forecast_result['forecast']
            confidence_intervals = forecast_result['confidence_intervals']
            
            logger.info(f"Forecast values: {forecast.values}")
            logger.info(f"Forecast shape: {forecast.shape}")
            logger.info(f"Confidence intervals shape: {confidence_intervals.shape}")
            logger.info(f"Confidence intervals head: {confidence_intervals.head()}")
            
            # Check if forecast is flat
            if len(forecast) > 1:
                forecast_diff = np.diff(forecast.values)
                logger.info(f"Forecast differences: {forecast_diff}")
                logger.info(f"Forecast is flat: {np.allclose(forecast_diff, 0, atol=1e-6)}")
            
        except Exception as e:
            logger.error(f"Error testing {indicator}: {e}")
            import traceback
            logger.error(traceback.format_exc())

if __name__ == "__main__":
    debug_forecasting() 