#!/usr/bin/env python3
"""
Test script to debug ARIMA model and see why it's producing flat forecasts
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

def test_arima_forecasting():
    """Test ARIMA forecasting specifically"""
    
    # Initialize FRED data collector
    api_key = os.getenv('FRED_API_KEY')
    if not api_key:
        logger.error("FRED_API_KEY not found in environment")
        return
    
    collector = FREDDataCollectorV2(api_key)
    
    # Fetch data
    indicators = ['GDPC1', 'INDPRO', 'RSAFS']
    data = collector.get_economic_data(indicators)
    df = collector.create_dataframe(data)
    
    # Create forecaster
    forecaster = EconomicForecaster(df)
    
    # Test GDPC1 specifically
    indicator = 'GDPC1'
    logger.info(f"\n=== Testing {indicator} ===")
    
    # Get raw data for ARIMA
    raw_series = forecaster.prepare_data(indicator, for_arima=True)
    logger.info(f"Raw series shape: {raw_series.shape}")
    logger.info(f"Raw series head: {raw_series.head()}")
    logger.info(f"Raw series tail: {raw_series.tail()}")
    logger.info(f"Raw series stats: mean={raw_series.mean():.2f}, std={raw_series.std():.2f}")
    logger.info(f"Raw series range: {raw_series.min():.2f} to {raw_series.max():.2f}")
    
    # Test ARIMA fitting
    try:
        model = forecaster.fit_arima_model(raw_series)
        logger.info(f"ARIMA model fitted successfully: {model}")
        
        # Test forecasting
        forecast_result = forecaster.forecast_series(raw_series, model_type='arima')
        forecast = forecast_result['forecast']
        confidence_intervals = forecast_result['confidence_intervals']
        
        logger.info(f"Forecast values: {forecast.tolist()}")
        logger.info(f"Forecast differences: {[forecast.iloc[i] - forecast.iloc[i-1] for i in range(1, len(forecast))]}")
        logger.info(f"Forecast is flat: {len(set(forecast)) == 1}")
        
        # Check if forecast is flat
        if len(set(forecast)) == 1:
            logger.warning("FORECAST IS FLAT!")
            # Try to understand why
            logger.info(f"Model AIC: {model.aic}")
            logger.info(f"Model parameters: {model.params}")
            
            # Check if the series has enough variation
            series_std = raw_series.std()
            series_range = raw_series.max() - raw_series.min()
            logger.info(f"Series std: {series_std}")
            logger.info(f"Series range: {series_range}")
            
            if series_std < 1e-6:
                logger.error("Series has almost no variation!")
            elif series_range < 1e-6:
                logger.error("Series has almost no range!")
            else:
                logger.info("Series has variation, but ARIMA is still producing flat forecast")
                
        else:
            logger.info("Forecast is NOT flat - working correctly!")
            
    except Exception as e:
        logger.error(f"ARIMA test failed: {e}")

if __name__ == "__main__":
    test_arima_forecasting() 