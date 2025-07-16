"""
Mathematical Fixes Module
Addresses key mathematical issues in economic data analysis:
1. Unit normalization and scaling
2. Frequency alignment and resampling
3. Correct growth rate calculation
4. Stationarity enforcement
5. Forecast period scaling
6. Safe error metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class MathematicalFixes:
    """
    Comprehensive mathematical fixes for economic data analysis
    """
    
    def __init__(self):
        """Initialize mathematical fixes"""
        self.frequency_map = {
            'D': 30,  # Daily -> 30 periods per quarter
            'M': 3,   # Monthly -> 3 periods per quarter  
            'Q': 1    # Quarterly -> 1 period per quarter
        }
        
        # Unit normalization factors - CORRECTED based on actual FRED data
        self.unit_factors = {
            'GDPC1': 1,         # FRED GDPC1 is already in correct units (billions)
            'INDPRO': 1,       # Index, no change
            'RSAFS': 1e3,      # FRED RSAFS is in millions, convert to billions
            'CPIAUCSL': 1,     # Index, no change (should be ~316, not 21.9)
            'FEDFUNDS': 1,     # Percent, no change
            'DGS10': 1,        # Percent, no change
            'UNRATE': 1,       # Percent, no change
            'PAYEMS': 1e3,     # Convert to thousands
            'PCE': 1e9,        # Convert to billions
            'M2SL': 1e9,       # Convert to billions
            'TCU': 1,          # Percent, no change
            'DEXUSEU': 1       # Exchange rate, no change
        }
    
    def normalize_units(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize units across all economic indicators
        
        Args:
            data: DataFrame with economic indicators
            
        Returns:
            DataFrame with normalized units
        """
        logger.info("Normalizing units across economic indicators")
        
        normalized_data = data.copy()
        
        for column in data.columns:
            if column in self.unit_factors:
                factor = self.unit_factors[column]
                if factor != 1:  # Only convert if factor is not 1
                    normalized_data[column] = data[column] * factor
                    logger.debug(f"Normalized {column} by factor {factor}")
                else:
                    # Keep original values for factors of 1
                    normalized_data[column] = data[column]
                    logger.debug(f"Kept {column} as original value")
        
        return normalized_data
    
    def align_frequencies(self, data: pd.DataFrame, target_freq: str = 'Q') -> pd.DataFrame:
        """
        Align all series to a common frequency
        
        Args:
            data: DataFrame with economic indicators
            target_freq: Target frequency ('D', 'M', 'Q')
            
        Returns:
            DataFrame with aligned frequencies
        """
        logger.info(f"Aligning frequencies to {target_freq}")
        
        aligned_data = pd.DataFrame()
        
        for column in data.columns:
            series = data[column].dropna()
            
            if not series.empty:
                # Resample to target frequency
                if target_freq == 'Q':
                    # For quarterly, use mean for most series, last value for rates
                    if column in ['FEDFUNDS', 'DGS10', 'UNRATE', 'TCU']:
                        resampled = series.resample('QE').last()
                    else:
                        resampled = series.resample('QE').mean()
                elif target_freq == 'M':
                    # For monthly, use mean for most series, last value for rates
                    if column in ['FEDFUNDS', 'DGS10', 'UNRATE', 'TCU']:
                        resampled = series.resample('ME').last()
                    else:
                        resampled = series.resample('ME').mean()
                else:
                    # For daily, forward fill
                    resampled = series.resample('D').ffill()
                
                aligned_data[column] = resampled
        
        return aligned_data
    
    def calculate_growth_rates(self, data: pd.DataFrame, method: str = 'pct_change') -> pd.DataFrame:
        """
        Calculate growth rates with proper handling
        
        Args:
            data: DataFrame with economic indicators
            method: Method for growth calculation ('pct_change', 'log_diff')
            
        Returns:
            DataFrame with growth rates
        """
        logger.info(f"Calculating growth rates using {method} method")
        
        growth_data = pd.DataFrame()
        
        for column in data.columns:
            series = data[column].dropna()
            
            if len(series) > 1:
                if method == 'pct_change':
                    # Calculate percent change
                    growth = series.pct_change() * 100
                elif method == 'log_diff':
                    # Calculate log difference
                    growth = np.log(series / series.shift(1)) * 100
                else:
                    # Default to percent change
                    growth = series.pct_change() * 100
                
                growth_data[column] = growth
        
        return growth_data
    
    def enforce_stationarity(self, data: pd.DataFrame, max_diffs: int = 2) -> Tuple[pd.DataFrame, Dict]:
        """
        Enforce stationarity through differencing
        
        Args:
            data: DataFrame with economic indicators
            max_diffs: Maximum number of differences to apply
            
        Returns:
            Tuple of (stationary_data, differencing_info)
        """
        logger.info("Enforcing stationarity through differencing")
        
        stationary_data = pd.DataFrame()
        differencing_info = {}
        
        for column in data.columns:
            series = data[column].dropna()
            
            if len(series) > 1:
                # Apply differencing until stationary
                diff_count = 0
                current_series = series
                
                while diff_count < max_diffs:
                    # Simple stationarity check (can be enhanced with ADF test)
                    if self._is_stationary(current_series):
                        break
                    
                    current_series = current_series.diff().dropna()
                    diff_count += 1
                
                stationary_data[column] = current_series
                differencing_info[column] = {
                    'diffs_applied': diff_count,
                    'is_stationary': self._is_stationary(current_series)
                }
        
        return stationary_data, differencing_info
    
    def _is_stationary(self, series: pd.Series, threshold: float = 0.05) -> bool:
        """
        Simple stationarity check based on variance
        
        Args:
            series: Time series to check
            threshold: Variance threshold for stationarity
            
        Returns:
            True if series appears stationary
        """
        if len(series) < 10:
            return True
        
        # Split series into halves and compare variance
        mid = len(series) // 2
        first_half = series[:mid]
        second_half = series[mid:]
        
        var_ratio = second_half.var() / first_half.var()
        
        # If variance ratio is close to 1, series is likely stationary
        return 0.5 <= var_ratio <= 2.0
    
    def scale_forecast_periods(self, forecast_periods: int, indicator: str, data: pd.DataFrame) -> int:
        """
        Scale forecast periods based on indicator frequency
        
        Args:
            forecast_periods: Base forecast periods
            indicator: Economic indicator name
            data: DataFrame with economic data
            
        Returns:
            Scaled forecast periods
        """
        if indicator not in data.columns:
            return forecast_periods
        
        series = data[indicator].dropna()
        if len(series) < 2:
            return forecast_periods
        
        # Determine frequency from data
        freq = self._infer_frequency(series)
        
        # Scale forecast periods
        if freq == 'D':
            return forecast_periods * 30  # 30 days per quarter
        elif freq == 'M':
            return forecast_periods * 3   # 3 months per quarter
        else:
            return forecast_periods        # Already quarterly
    
    def _infer_frequency(self, series: pd.Series) -> str:
        """
        Infer frequency from time series
        
        Args:
            series: Time series
            
        Returns:
            Frequency string ('D', 'M', 'Q')
        """
        if len(series) < 2:
            return 'Q'
        
        # Calculate average time difference
        time_diff = series.index.to_series().diff().dropna()
        avg_diff = time_diff.mean()
        
        if avg_diff.days <= 1:
            return 'D'
        elif avg_diff.days <= 35:
            return 'M'
        else:
            return 'Q'
    
    def safe_mape(self, actual: np.ndarray, forecast: np.ndarray) -> float:
        """
        Calculate safe MAPE with protection against division by zero
        
        Args:
            actual: Actual values
            forecast: Forecasted values
            
        Returns:
            MAPE value
        """
        actual = np.array(actual)
        forecast = np.array(forecast)
        
        # Avoid division by zero
        denominator = np.maximum(np.abs(actual), 1e-8)
        mape = np.mean(np.abs((actual - forecast) / denominator)) * 100
        
        return mape
    
    def safe_mae(self, actual: np.ndarray, forecast: np.ndarray) -> float:
        """
        Calculate MAE (Mean Absolute Error)
        
        Args:
            actual: Actual values
            forecast: Forecasted values
            
        Returns:
            MAE value
        """
        actual = np.array(actual)
        forecast = np.array(forecast)
        
        return np.mean(np.abs(actual - forecast))
    
    def safe_rmse(self, actual: np.ndarray, forecast: np.ndarray) -> float:
        """Calculate RMSE safely handling edge cases"""
        if len(actual) == 0 or len(forecast) == 0:
            return np.inf
        
        # Ensure same length
        min_len = min(len(actual), len(forecast))
        if min_len == 0:
            return np.inf
        
        actual_trimmed = actual[:min_len]
        forecast_trimmed = forecast[:min_len]
        
        # Remove any infinite or NaN values
        mask = np.isfinite(actual_trimmed) & np.isfinite(forecast_trimmed)
        if not np.any(mask):
            return np.inf
        
        actual_clean = actual_trimmed[mask]
        forecast_clean = forecast_trimmed[mask]
        
        if len(actual_clean) == 0:
            return np.inf
        
        return np.sqrt(np.mean((actual_clean - forecast_clean) ** 2))
    
    def validate_scaling(self, series: pd.Series,
                         unit_hint: str,
                         expected_min: float,
                         expected_max: float):
        """
        Checks if values fall within expected magnitude range.
        Args:
            series: pandas Series of numeric data.
            unit_hint: description, e.g., "Real GDP".
            expected_min / expected_max: plausible lower/upper bounds (same units).
        Raises:
            ValueError if data outside range for >5% of values.
        """
        vals = series.dropna()
        mask = (vals < expected_min) | (vals > expected_max)
        if mask.mean() > 0.05:
            raise ValueError(f"{unit_hint}: {mask.mean():.1%} of data "
                             f"outside [{expected_min}, {expected_max}]. "
                             "Check for scaling/unit issues.")
        print(f"{unit_hint}: data within expected range.")
    
    def apply_comprehensive_fixes(self, data: pd.DataFrame, 
                                target_freq: str = 'Q',
                                growth_method: str = 'pct_change',
                                normalize_units: bool = True,
                                preserve_absolute_values: bool = False) -> Tuple[pd.DataFrame, Dict]:
        """
        Apply comprehensive mathematical fixes to economic data
        
        Args:
            data: DataFrame with economic indicators
            target_freq: Target frequency ('D', 'M', 'Q')
            growth_method: Method for growth calculation ('pct_change', 'log_diff')
            normalize_units: Whether to normalize units
            preserve_absolute_values: Whether to preserve absolute values for display
            
        Returns:
            Tuple of (processed_data, fix_info)
        """
        logger.info("Applying comprehensive mathematical fixes")
        
        fix_info = {
            'original_shape': data.shape,
            'frequency_alignment': {},
            'unit_normalization': {},
            'growth_calculation': {},
            'stationarity_enforcement': {},
            'validation_results': {}
        }
        
        processed_data = data.copy()
        
        # Step 1: Align frequencies
        if target_freq != 'auto':
            processed_data = self.align_frequencies(processed_data, target_freq)
            fix_info['frequency_alignment'] = {
                'target_frequency': target_freq,
                'final_shape': processed_data.shape
            }
        
        # Step 2: Normalize units
        if normalize_units:
            processed_data = self.normalize_units(processed_data)
            fix_info['unit_normalization'] = {
                'normalized_indicators': list(processed_data.columns)
            }
        
        # Step 3: Calculate growth rates if requested
        if growth_method in ['pct_change', 'log_diff']:
            growth_data = self.calculate_growth_rates(processed_data, growth_method)
            fix_info['growth_calculation'] = {
                'method': growth_method,
                'growth_indicators': list(growth_data.columns)
            }
            # For now, keep both absolute and growth data
            if not preserve_absolute_values:
                processed_data = growth_data
        
        # Step 4: Enforce stationarity
        stationary_data, differencing_info = self.enforce_stationarity(processed_data)
        fix_info['stationarity_enforcement'] = differencing_info
        
        # Step 5: Validate processed data
        validation_results = self._validate_processed_data(processed_data)
        fix_info['validation_results'] = validation_results
        
        logger.info(f"Comprehensive fixes applied. Final shape: {processed_data.shape}")
        return processed_data, fix_info
    
    def _validate_processed_data(self, data: pd.DataFrame) -> Dict:
        """
        Validate processed data for scaling and quality issues
        
        Args:
            data: Processed DataFrame
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'scaling_issues': [],
            'quality_warnings': [],
            'validation_score': 100.0
        }
        
        for column in data.columns:
            series = data[column].dropna()
            
            if len(series) == 0:
                validation_results['quality_warnings'].append(f"{column}: No data available")
                continue
            
            # Check for extreme values that might indicate scaling issues
            mean_val = series.mean()
            std_val = series.std()
            
            # Check for values that are too large or too small
            if abs(mean_val) > 1e6:
                validation_results['scaling_issues'].append(
                    f"{column}: Mean value {mean_val:.2e} is extremely large - possible scaling issue"
                )
            
            if std_val > 1e5:
                validation_results['scaling_issues'].append(
                    f"{column}: Standard deviation {std_val:.2e} is extremely large - possible scaling issue"
                )
            
            # Check for values that are too close to zero (might indicate unit conversion issues)
            if abs(mean_val) < 1e-6 and std_val < 1e-6:
                validation_results['scaling_issues'].append(
                    f"{column}: Values are extremely small - possible unit conversion issue"
                )
        
        # Calculate validation score
        total_checks = len(data.columns)
        failed_checks = len(validation_results['scaling_issues']) + len(validation_results['quality_warnings'])
        
        if total_checks > 0:
            validation_results['validation_score'] = max(0, 100 - (failed_checks / total_checks) * 100)
        
        return validation_results 