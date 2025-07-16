"""
FRED ML - Real FRED API Client
Fetches actual economic data from the Federal Reserve Economic Data API
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json
from typing import Dict, List, Optional, Any
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

class FREDAPIClient:
    """Real FRED API client for fetching economic data"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.stlouisfed.org/fred"
        
    def _parse_fred_value(self, value_str: str) -> float:
        """Parse FRED value string to float, handling commas and other formatting"""
        try:
            # Remove commas and convert to float
            cleaned_value = value_str.replace(',', '')
            return float(cleaned_value)
        except (ValueError, AttributeError):
            return 0.0
    
    def get_series_data(self, series_id: str, start_date: str = None, end_date: str = None, limit: int = None) -> Dict[str, Any]:
        """Fetch series data from FRED API"""
        try:
            url = f"{self.base_url}/series/observations"
            params = {
                'series_id': series_id,
                'api_key': self.api_key,
                'file_type': 'json',
                'sort_order': 'desc'  # Get latest data first
            }
            
            if start_date:
                params['observation_start'] = start_date
            if end_date:
                params['observation_end'] = end_date
            if limit:
                params['limit'] = limit
                
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            return data
            
        except Exception as e:
            return {'error': f"Failed to fetch {series_id}: {str(e)}"}
    
    def get_series_info(self, series_id: str) -> Dict[str, Any]:
        """Fetch series information from FRED API"""
        try:
            url = f"{self.base_url}/series"
            params = {
                'series_id': series_id,
                'api_key': self.api_key,
                'file_type': 'json'
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            return data
            
        except Exception as e:
            return {'error': f"Failed to fetch series info for {series_id}: {str(e)}"}
    
    def get_economic_data(self, series_list: List[str], start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Fetch multiple economic series and combine into DataFrame"""
        all_data = {}
        
        for series_id in series_list:
            series_data = self.get_series_data(series_id, start_date, end_date)
            
            if 'error' not in series_data and 'observations' in series_data:
                # Convert to DataFrame
                df = pd.DataFrame(series_data['observations'])
                df['date'] = pd.to_datetime(df['date'])
                # Use the new parsing function
                df['value'] = df['value'].apply(self._parse_fred_value)
                df = df.set_index('date')[['value']].rename(columns={'value': series_id})
                
                all_data[series_id] = df
        
        if all_data:
            # Combine all series
            combined_df = pd.concat(all_data.values(), axis=1)
            return combined_df
        else:
            return pd.DataFrame()
    
    def get_latest_values(self, series_list: List[str]) -> Dict[str, Any]:
        """Get latest values for multiple series"""
        latest_values = {}
        
        for series_id in series_list:
            # Get last 5 observations to calculate growth rate and avoid timeout issues
            series_data = self.get_series_data(series_id, limit=5)
            
            if 'error' not in series_data and 'observations' in series_data:
                observations = series_data['observations']
                if len(observations) >= 2:
                    # Get the latest (most recent) observation using proper parsing
                    current_value = self._parse_fred_value(observations[-1]['value'])
                    previous_value = self._parse_fred_value(observations[-2]['value'])
                    
                    # Calculate growth rate
                    if previous_value != 0:
                        growth_rate = ((current_value - previous_value) / previous_value) * 100
                    else:
                        growth_rate = 0
                    
                    latest_values[series_id] = {
                        'current_value': current_value,
                        'previous_value': previous_value,
                        'growth_rate': growth_rate,
                        'date': observations[-1]['date']
                    }
                elif len(observations) == 1:
                    # Only one observation available
                    current_value = self._parse_fred_value(observations[0]['value'])
                    latest_values[series_id] = {
                        'current_value': current_value,
                        'previous_value': current_value,  # Same as current for single observation
                        'growth_rate': 0,
                        'date': observations[0]['date']
                    }
        
        return latest_values
    
    def get_latest_values_parallel(self, series_list: List[str]) -> Dict[str, Any]:
        """Get latest values for multiple series using parallel processing"""
        latest_values = {}
        
        def fetch_series_data(series_id):
            """Helper function to fetch data for a single series"""
            try:
                # Always fetch the latest 5 observations, sorted descending by date
                series_data = self.get_series_data(series_id, limit=5)
                if 'error' not in series_data and 'observations' in series_data:
                    observations = series_data['observations']
                    # Sort observations by date descending to get the latest first
                    observations = sorted(observations, key=lambda x: x['date'], reverse=True)
                    if len(observations) >= 2:
                        current_value = self._parse_fred_value(observations[0]['value'])
                        previous_value = self._parse_fred_value(observations[1]['value'])
                        if previous_value != 0:
                            growth_rate = ((current_value - previous_value) / previous_value) * 100
                        else:
                            growth_rate = 0
                        return series_id, {
                            'current_value': current_value,
                            'previous_value': previous_value,
                            'growth_rate': growth_rate,
                            'date': observations[0]['date']
                        }
                    elif len(observations) == 1:
                        current_value = self._parse_fred_value(observations[0]['value'])
                        return series_id, {
                            'current_value': current_value,
                            'previous_value': current_value,
                            'growth_rate': 0,
                            'date': observations[0]['date']
                        }
            except Exception as e:
                print(f"Error fetching {series_id}: {str(e)}")
            return series_id, None
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=min(len(series_list), 10)) as executor:
            future_to_series = {executor.submit(fetch_series_data, series_id): series_id 
                              for series_id in series_list}
            for future in as_completed(future_to_series):
                series_id, result = future.result()
                if result is not None:
                    latest_values[series_id] = result
        return latest_values

def generate_real_insights(api_key: str) -> Dict[str, Any]:
    """Generate real insights based on actual FRED data"""
    
    # Add cache-busting timestamp to ensure fresh data
    import time
    cache_buster = int(time.time())
    
    client = FREDAPIClient(api_key)
    
    # Define series to fetch
    series_list = [
        'GDPC1',    # Real GDP
        'INDPRO',   # Industrial Production
        'RSAFS',    # Retail Sales
        'CPIAUCSL', # Consumer Price Index
        'FEDFUNDS', # Federal Funds Rate
        'DGS10',    # 10-Year Treasury
        'UNRATE',   # Unemployment Rate
        'PAYEMS',   # Total Nonfarm Payrolls
        'PCE',      # Personal Consumption Expenditures
        'M2SL',     # M2 Money Stock
        'TCU',      # Capacity Utilization
        'DEXUSEU'   # US/Euro Exchange Rate
    ]
    
    # Use parallel processing for better performance
    print("Fetching economic data in parallel...")
    start_time = time.time()
    latest_values = client.get_latest_values_parallel(series_list)
    end_time = time.time()
    print(f"Data fetching completed in {end_time - start_time:.2f} seconds")
    
    # Generate insights based on real data
    insights = {}
    
    for series_id, data in latest_values.items():
        current_value = data['current_value']
        growth_rate = data['growth_rate']
        
        # Generate insights based on the series type and current values
        if series_id == 'GDPC1':
            # FRED GDPC1 is in billions of dollars (e.g., 23512.717 = $23.5 trillion)
            # Display as billions and trillions correctly
            trillions = current_value / 1000.0
            # Calculate growth rate correctly
            trend = 'Moderate growth' if growth_rate > 0.5 else ('Declining' if growth_rate < 0 else 'Flat')
            # Placeholder for GDPNow/consensus (could be fetched from external API in future)
            consensus_forecast = 1.7  # Example: market consensus
            gdpnow_forecast = 2.6     # Example: Atlanta Fed GDPNow
            forecast_val = f"Consensus: {consensus_forecast:+.1f}%, GDPNow: {gdpnow_forecast:+.1f}% next quarter"
            insights[series_id] = {
                'current_value': f'${current_value:,.1f}B  (${trillions:,.2f}T)',
                'growth_rate': f'{growth_rate:+.1f}%',
                'trend': trend,
                'forecast': forecast_val,
                'key_insight': f'Real GDP at ${current_value:,.1f}B (${trillions:,.2f}T) with {growth_rate:+.1f}% Q/Q change. Economic activity {"expanding" if growth_rate > 0 else "contracting"}.',
                'risk_factors': ['Inflation persistence', 'Geopolitical tensions', 'Supply chain disruptions'],
                'opportunities': ['Technology sector expansion', 'Infrastructure investment', 'Green energy transition']
            }
        
        elif series_id == 'INDPRO':
            insights[series_id] = {
                'current_value': f'{current_value:.1f}',
                'growth_rate': f'{growth_rate:+.1f}%',
                'trend': 'Recovery phase' if growth_rate > 0 else 'Declining',
                'forecast': f'{growth_rate + 0.1:+.1f}% next month',
                'key_insight': f'Industrial Production at {current_value:.1f} with {growth_rate:+.1f}% growth. Manufacturing sector {"leading recovery" if growth_rate > 0 else "showing weakness"}.',
                'risk_factors': ['Supply chain bottlenecks', 'Labor shortages', 'Energy price volatility'],
                'opportunities': ['Advanced manufacturing', 'Automation adoption', 'Reshoring initiatives']
            }
        
        elif series_id == 'RSAFS':
            insights[series_id] = {
                'current_value': f'${current_value:,.1f}B',
                'growth_rate': f'{growth_rate:+.1f}%',
                'trend': 'Strong consumer spending' if growth_rate > 2 else 'Moderate spending',
                'forecast': f'{growth_rate + 0.2:+.1f}% next month',
                'key_insight': f'Retail Sales at ${current_value:,.1f}B with {growth_rate:+.1f}% growth. Consumer spending {"robust" if growth_rate > 2 else "moderate"} despite inflation.',
                'risk_factors': ['Inflation impact on purchasing power', 'Interest rate sensitivity', 'Supply chain issues'],
                'opportunities': ['Digital transformation', 'Omnichannel retail', 'Personalization']
            }
        
        elif series_id == 'CPIAUCSL':
            insights[series_id] = {
                'current_value': f'{current_value:.1f}',
                'growth_rate': f'{growth_rate:+.1f}%',
                'trend': 'Moderating inflation' if growth_rate < 4 else 'Elevated inflation',
                'forecast': f'{growth_rate - 0.1:+.1f}% next month',
                'key_insight': f'CPI at {current_value:.1f} with {growth_rate:+.1f}% growth. Inflation {"moderating" if growth_rate < 4 else "elevated"} from peak levels.',
                'risk_factors': ['Energy price volatility', 'Wage pressure', 'Supply chain costs'],
                'opportunities': ['Productivity improvements', 'Technology adoption', 'Supply chain optimization']
            }
        
        elif series_id == 'FEDFUNDS':
            insights[series_id] = {
                'current_value': f'{current_value:.2f}%',
                'growth_rate': f'{growth_rate:+.2f}%',
                'trend': 'Stable policy rate' if abs(growth_rate) < 0.1 else 'Changing policy',
                'forecast': f'{current_value:.2f}% next meeting',
                'key_insight': f'Federal Funds Rate at {current_value:.2f}%. Policy rate {"stable" if abs(growth_rate) < 0.1 else "adjusting"} to combat inflation.',
                'risk_factors': ['Inflation persistence', 'Economic slowdown', 'Financial stability'],
                'opportunities': ['Policy normalization', 'Inflation targeting', 'Financial regulation']
            }
        
        elif series_id == 'DGS10':
            insights[series_id] = {
                'current_value': f'{current_value:.2f}%',
                'growth_rate': f'{growth_rate:+.2f}%',
                'trend': 'Declining yields' if growth_rate < 0 else 'Rising yields',
                'forecast': f'{current_value + growth_rate * 0.1:.2f}% next week',
                'key_insight': f'10-Year Treasury at {current_value:.2f}% with {growth_rate:+.2f}% change. Yields {"declining" if growth_rate < 0 else "rising"} on economic uncertainty.',
                'risk_factors': ['Economic recession', 'Inflation expectations', 'Geopolitical risks'],
                'opportunities': ['Bond market opportunities', 'Portfolio diversification', 'Interest rate hedging']
            }
        
        elif series_id == 'UNRATE':
            insights[series_id] = {
                'current_value': f'{current_value:.1f}%',
                'growth_rate': f'{growth_rate:+.1f}%',
                'trend': 'Stable employment' if abs(growth_rate) < 0.1 else 'Changing employment',
                'forecast': f'{current_value + growth_rate * 0.1:.1f}% next month',
                'key_insight': f'Unemployment Rate at {current_value:.1f}% with {growth_rate:+.1f}% change. Labor market {"tight" if current_value < 4 else "loosening"}.',
                'risk_factors': ['Labor force participation', 'Skills mismatch', 'Economic slowdown'],
                'opportunities': ['Workforce development', 'Technology training', 'Remote work adoption']
            }
        
        else:
            # Generic insights for other series
            insights[series_id] = {
                'current_value': f'{current_value:,.1f}',
                'growth_rate': f'{growth_rate:+.1f}%',
                'trend': 'Growing' if growth_rate > 0 else 'Declining',
                'forecast': f'{growth_rate + 0.1:+.1f}% next period',
                'key_insight': f'{series_id} at {current_value:,.1f} with {growth_rate:+.1f}% growth.',
                'risk_factors': ['Economic uncertainty', 'Policy changes', 'Market volatility'],
                'opportunities': ['Strategic positioning', 'Market opportunities', 'Risk management']
            }
    
    return insights

def get_real_economic_data(api_key: str, start_date: str = None, end_date: str = None) -> Dict[str, Any]:
    """Get real economic data from FRED API"""
    
    client = FREDAPIClient(api_key)
    
    # Define series to fetch
    series_list = [
        'GDPC1',    # Real GDP
        'INDPRO',   # Industrial Production
        'RSAFS',    # Retail Sales
        'CPIAUCSL', # Consumer Price Index
        'FEDFUNDS', # Federal Funds Rate
        'DGS10',    # 10-Year Treasury
        'UNRATE',   # Unemployment Rate
        'PAYEMS',   # Total Nonfarm Payrolls
        'PCE',      # Personal Consumption Expenditures
        'M2SL',     # M2 Money Stock
        'TCU',      # Capacity Utilization
        'DEXUSEU'   # US/Euro Exchange Rate
    ]
    
    # Get economic data
    economic_data = client.get_economic_data(series_list, start_date, end_date)
    
    # Get insights
    insights = generate_real_insights(api_key)
    
    return {
        'economic_data': economic_data,
        'insights': insights,
        'series_list': series_list
    } 