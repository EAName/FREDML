"""
Enhanced FRED Client
Advanced data collection for comprehensive economic indicators
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union

import pandas as pd
from fredapi import Fred

logger = logging.getLogger(__name__)

class EnhancedFREDClient:
    """
    Enhanced FRED API client for comprehensive economic data collection
    with support for multiple frequencies and advanced data processing
    """
    
    # Economic indicators mapping
    ECONOMIC_INDICATORS = {
        # Output & Activity
        'GDPC1': 'Real Gross Domestic Product (chained 2012 dollars)',
        'INDPRO': 'Industrial Production Index',
        'RSAFS': 'Retail Sales',
        'TCU': 'Capacity Utilization',
        'PAYEMS': 'Total Nonfarm Payrolls',
        
        # Prices & Inflation
        'CPIAUCSL': 'Consumer Price Index for All Urban Consumers',
        'PCE': 'Personal Consumption Expenditures',
        
        # Financial & Monetary
        'FEDFUNDS': 'Federal Funds Rate',
        'DGS10': '10-Year Treasury Rate',
        'M2SL': 'M2 Money Stock',
        
        # International
        'DEXUSEU': 'US/Euro Exchange Rate',
        
        # Labor
        'UNRATE': 'Unemployment Rate'
    }
    
    def __init__(self, api_key: str):
        """
        Initialize enhanced FRED client
        
        Args:
            api_key: FRED API key
        """
        self.fred = Fred(api_key=api_key)
        self.data_cache = {}
        
    def fetch_economic_data(self, indicators: List[str] = None, 
                          start_date: str = '1990-01-01',
                          end_date: str = None,
                          frequency: str = 'auto') -> pd.DataFrame:
        """
        Fetch comprehensive economic data
        
        Args:
            indicators: List of indicators to fetch. If None, fetch all available
            start_date: Start date for data collection
            end_date: End date for data collection. If None, use current date
            frequency: Data frequency ('auto', 'M', 'Q', 'A')
            
        Returns:
            DataFrame with economic indicators
        """
        if indicators is None:
            indicators = list(self.ECONOMIC_INDICATORS.keys())
        
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        logger.info(f"Fetching economic data for {len(indicators)} indicators")
        logger.info(f"Date range: {start_date} to {end_date}")
        
        data_dict = {}
        
        for indicator in indicators:
            try:
                if indicator in self.ECONOMIC_INDICATORS:
                    series_data = self._fetch_series(indicator, start_date, end_date, frequency)
                    if series_data is not None and not series_data.empty:
                        data_dict[indicator] = series_data
                        logger.info(f"Successfully fetched {indicator}: {len(series_data)} observations")
                    else:
                        logger.warning(f"No data available for {indicator}")
                else:
                    logger.warning(f"Unknown indicator: {indicator}")
                    
            except Exception as e:
                logger.error(f"Failed to fetch {indicator}: {e}")
        
        if not data_dict:
            raise ValueError("No data could be fetched for any indicators")
        
        # Combine all series into a single DataFrame
        combined_data = pd.concat(data_dict.values(), axis=1)
        combined_data.columns = list(data_dict.keys())
        
        # Sort by date
        combined_data = combined_data.sort_index()
        
        logger.info(f"Combined data shape: {combined_data.shape}")
        logger.info(f"Date range: {combined_data.index.min()} to {combined_data.index.max()}")
        
        return combined_data
    
    def _fetch_series(self, series_id: str, start_date: str, end_date: str, 
                     frequency: str) -> Optional[pd.Series]:
        """
        Fetch individual series with frequency handling
        
        Args:
            series_id: FRED series ID
            start_date: Start date
            end_date: End date
            frequency: Data frequency (for post-processing)
            
        Returns:
            Series data or None if failed
        """
        try:
            # Fetch data without frequency parameter (FRED API doesn't support it)
            series = self.fred.get_series(
                series_id,
                observation_start=start_date,
                observation_end=end_date
            )
            
            if series.empty:
                logger.warning(f"No data returned for {series_id}")
                return None
            
            # Handle frequency conversion if needed
            if frequency == 'auto':
                series = self._standardize_frequency(series, series_id)
            elif frequency == 'Q':
                # Convert to quarterly if requested
                series = self._convert_to_quarterly(series, series_id)
            elif frequency == 'M':
                # Convert to monthly if requested
                series = self._convert_to_monthly(series, series_id)
            
            return series
            
        except Exception as e:
            logger.error(f"Error fetching {series_id}: {e}")
            return None
    
    def _convert_to_quarterly(self, series: pd.Series, series_id: str) -> pd.Series:
        """Convert series to quarterly frequency"""
        if series_id in ['INDPRO', 'RSAFS', 'TCU', 'PAYEMS', 'CPIAUCSL', 'M2SL']:
            return series.resample('Q').last()
        else:
            return series.resample('Q').mean()
    
    def _convert_to_monthly(self, series: pd.Series, series_id: str) -> pd.Series:
        """Convert series to monthly frequency"""
        return series.resample('M').last()
    
    def _get_appropriate_frequency(self, series_id: str) -> str:
        """
        Get appropriate frequency for a series based on its characteristics
        
        Args:
            series_id: FRED series ID
            
        Returns:
            Appropriate frequency string
        """
        # Quarterly series
        quarterly_series = ['GDPC1', 'PCE']
        
        # Monthly series (most common)
        monthly_series = ['INDPRO', 'RSAFS', 'TCU', 'PAYEMS', 'CPIAUCSL', 
                         'FEDFUNDS', 'DGS10', 'M2SL', 'DEXUSEU', 'UNRATE']
        
        if series_id in quarterly_series:
            return 'Q'
        elif series_id in monthly_series:
            return 'M'
        else:
            return 'M'  # Default to monthly
    
    def _standardize_frequency(self, series: pd.Series, series_id: str) -> pd.Series:
        """
        Standardize frequency for consistent analysis
        
        Args:
            series: Time series data
            series_id: Series ID for context
            
        Returns:
            Standardized series
        """
        # For quarterly analysis, convert monthly to quarterly
        if series_id in ['INDPRO', 'RSAFS', 'TCU', 'PAYEMS', 'CPIAUCSL', 
                        'FEDFUNDS', 'DGS10', 'M2SL', 'DEXUSEU', 'UNRATE']:
            # Use end-of-quarter values for most series
            if series_id in ['INDPRO', 'RSAFS', 'TCU', 'PAYEMS', 'CPIAUCSL', 'M2SL']:
                return series.resample('Q').last()
            else:
                # For rates, use mean
                return series.resample('Q').mean()
        
        return series
    
    def fetch_quarterly_data(self, indicators: List[str] = None,
                           start_date: str = '1990-01-01',
                           end_date: str = None) -> pd.DataFrame:
        """
        Fetch data standardized to quarterly frequency
        
        Args:
            indicators: List of indicators to fetch
            start_date: Start date
            end_date: End date
            
        Returns:
            Quarterly DataFrame
        """
        return self.fetch_economic_data(indicators, start_date, end_date, frequency='Q')
    
    def fetch_monthly_data(self, indicators: List[str] = None,
                          start_date: str = '1990-01-01',
                          end_date: str = None) -> pd.DataFrame:
        """
        Fetch data standardized to monthly frequency
        
        Args:
            indicators: List of indicators to fetch
            start_date: Start date
            end_date: End date
            
        Returns:
            Monthly DataFrame
        """
        return self.fetch_economic_data(indicators, start_date, end_date, frequency='M')
    
    def get_series_info(self, series_id: str) -> Dict:
        """
        Get detailed information about a series
        
        Args:
            series_id: FRED series ID
            
        Returns:
            Dictionary with series information
        """
        try:
            info = self.fred.get_series_info(series_id)
            return {
                'id': info.id,
                'title': info.title,
                'units': info.units,
                'frequency': info.frequency,
                'seasonal_adjustment': info.seasonal_adjustment,
                'last_updated': info.last_updated,
                'notes': info.notes
            }
        except Exception as e:
            logger.error(f"Failed to get info for {series_id}: {e}")
            return {'error': str(e)}
    
    def get_all_series_info(self, indicators: List[str] = None) -> Dict:
        """
        Get information for all indicators
        
        Args:
            indicators: List of indicators. If None, use all available
            
        Returns:
            Dictionary with series information
        """
        if indicators is None:
            indicators = list(self.ECONOMIC_INDICATORS.keys())
        
        series_info = {}
        
        for indicator in indicators:
            if indicator in self.ECONOMIC_INDICATORS:
                info = self.get_series_info(indicator)
                series_info[indicator] = info
                logger.info(f"Retrieved info for {indicator}")
        
        return series_info
    
    def validate_data_quality(self, data: pd.DataFrame) -> Dict:
        """
        Validate data quality and check for common issues
        
        Args:
            data: DataFrame with economic indicators
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'missing_data': {},
            'outliers': {},
            'data_quality_score': 0.0,
            'warnings': [],
            'errors': []
        }
        
        total_series = len(data.columns)
        valid_series = 0
        
        for column in data.columns:
            series = data[column].dropna()
            
            if len(series) == 0:
                validation_results['missing_data'][column] = 'No data available'
                validation_results['errors'].append(f"{column}: No data available")
                continue
            
            # Check for missing data
            missing_pct = (data[column].isna().sum() / len(data)) * 100
            if missing_pct > 20:
                validation_results['missing_data'][column] = f"{missing_pct:.1f}% missing"
                validation_results['warnings'].append(f"{column}: {missing_pct:.1f}% missing data")
            
            # Check for outliers using IQR method
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = series[(series < lower_bound) | (series > upper_bound)]
            outlier_pct = (len(outliers) / len(series)) * 100
            
            if outlier_pct > 5:
                validation_results['outliers'][column] = f"{outlier_pct:.1f}% outliers"
                validation_results['warnings'].append(f"{column}: {outlier_pct:.1f}% outliers detected")
            
            # Validate scaling for known indicators
            self._validate_economic_scaling(series, column, validation_results)
            
            valid_series += 1
        
        # Calculate overall data quality score
        if total_series > 0:
            validation_results['data_quality_score'] = (valid_series / total_series) * 100
        
        return validation_results
    
    def _validate_economic_scaling(self, series: pd.Series, indicator: str, validation_results: Dict):
        """
        Validate economic indicator scaling using expected ranges
        
        Args:
            series: Time series data
            indicator: Indicator name
            validation_results: Validation results dictionary to update
        """
        # Expected ranges for common economic indicators
        scaling_ranges = {
            'GDPC1': (15000, 25000),  # Real GDP in billions (2020-2024 range)
            'INDPRO': (90, 110),       # Industrial Production Index
            'CPIAUCSL': (250, 350),    # Consumer Price Index
            'FEDFUNDS': (0, 10),       # Federal Funds Rate (%)
            'DGS10': (0, 8),           # 10-Year Treasury Rate (%)
            'UNRATE': (3, 15),         # Unemployment Rate (%)
            'PAYEMS': (140000, 160000), # Total Nonfarm Payrolls (thousands)
            'PCE': (15000, 25000),     # Personal Consumption Expenditures (billions)
            'M2SL': (20000, 25000),    # M2 Money Stock (billions)
            'TCU': (60, 90),           # Capacity Utilization (%)
            'DEXUSEU': (0.8, 1.2),     # US/Euro Exchange Rate
            'RSAFS': (400000, 600000)  # Retail Sales (millions)
        }
        
        if indicator in scaling_ranges:
            expected_min, expected_max = scaling_ranges[indicator]
            
            # Check if values fall within expected range
            vals = series.dropna()
            if len(vals) > 0:
                mask = (vals < expected_min) | (vals > expected_max)
                outlier_pct = mask.mean() * 100
                
                if outlier_pct > 5:
                    validation_results['warnings'].append(
                        f"{indicator}: {outlier_pct:.1f}% of data outside expected range "
                        f"[{expected_min}, {expected_max}]. Check for scaling/unit issues."
                    )
                else:
                    logger.debug(f"{indicator}: data within expected range [{expected_min}, {expected_max}]")
    
    def generate_data_summary(self, data: pd.DataFrame) -> str:
        """
        Generate comprehensive data summary report
        
        Args:
            data: Economic data DataFrame
            
        Returns:
            Formatted summary report
        """
        quality_report = self.validate_data_quality(data)
        
        summary = "ECONOMIC DATA SUMMARY\n"
        summary += "=" * 50 + "\n\n"
        
        summary += f"Dataset Overview:\n"
        summary += f"  Total Series: {quality_report['total_series']}\n"
        summary += f"  Total Observations: {quality_report['total_observations']}\n"
        summary += f"  Date Range: {quality_report['date_range']['start']} to {quality_report['date_range']['end']}\n\n"
        
        summary += f"Series Information:\n"
        for indicator in data.columns:
            if indicator in self.ECONOMIC_INDICATORS:
                summary += f"  {indicator}: {self.ECONOMIC_INDICATORS[indicator]}\n"
        summary += "\n"
        
        summary += f"Data Quality:\n"
        for series, metrics in quality_report['missing_data'].items():
            summary += f"  {series}: {metrics['completeness']:.1f}% complete "
            summary += f"({metrics['missing_count']} missing observations)\n"
        
        summary += "\n"
        
        return summary 