#!/usr/bin/env python3
"""
FRED Data Collector v2
A tool for collecting and analyzing Federal Reserve Economic Data (FRED)
using direct API calls instead of the fredapi library
"""

import os
import warnings
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns

warnings.filterwarnings("ignore")

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from config.settings import (DEFAULT_END_DATE, DEFAULT_START_DATE,
                             FRED_API_KEY, OUTPUT_DIR, PLOTS_DIR)


class FREDDataCollectorV2:
    def __init__(self, api_key=None):
        """Initialize the FRED data collector with API key."""
        self.api_key = api_key or FRED_API_KEY
        self.base_url = "https://api.stlouisfed.org/fred"

        # Create output directories
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(PLOTS_DIR, exist_ok=True)

        # Common economic indicators
        self.indicators = {
            "GDP": "GDP",  # Gross Domestic Product
            "UNRATE": "UNRATE",  # Unemployment Rate
            "CPIAUCSL": "CPIAUCSL",  # Consumer Price Index
            "FEDFUNDS": "FEDFUNDS",  # Federal Funds Rate
            "DGS10": "DGS10",  # 10-Year Treasury Rate
            "DEXUSEU": "DEXUSEU",  # US/Euro Exchange Rate
            "PAYEMS": "PAYEMS",  # Total Nonfarm Payrolls
            "INDPRO": "INDPRO",  # Industrial Production
            "M2SL": "M2SL",  # M2 Money Stock
            "PCE": "PCE",  # Personal Consumption Expenditures
        }

    def get_series_info(self, series_id):
        """Get information about a FRED series."""
        try:
            url = f"{self.base_url}/series"
            params = {
                "series_id": series_id,
                "api_key": self.api_key,
                "file_type": "json",
            }

            response = requests.get(url, params=params)

            if response.status_code == 200:
                data = response.json()
                series = data.get("seriess", [])

                if series:
                    s = series[0]
                    return {
                        "id": s["id"],
                        "title": s["title"],
                        "units": s.get("units", ""),
                        "frequency": s.get("frequency", ""),
                        "last_updated": s.get("last_updated", ""),
                        "notes": s.get("notes", ""),
                    }

            return None

        except Exception as e:
            print(f"Error getting info for {series_id}: {e}")
            return None

    def get_economic_data(self, series_ids, start_date=None, end_date=None):
        """Fetch economic data for specified series."""
        start_date = start_date or DEFAULT_START_DATE
        end_date = end_date or DEFAULT_END_DATE

        data = {}

        for series_id in series_ids:
            try:
                print(f"Fetching data for {series_id}...")

                url = f"{self.base_url}/series/observations"
                params = {
                    "series_id": series_id,
                    "api_key": self.api_key,
                    "file_type": "json",
                    "start_date": start_date,
                    "end_date": end_date,
                }

                response = requests.get(url, params=params)

                if response.status_code == 200:
                    response_data = response.json()
                    observations = response_data.get("observations", [])

                    if observations:
                        # Convert to pandas Series
                        dates = []
                        values = []

                        for obs in observations:
                            try:
                                date = pd.to_datetime(obs["date"])
                                value = (
                                    float(obs["value"])
                                    if obs["value"] != "."
                                    else np.nan
                                )
                                dates.append(date)
                                values.append(value)
                            except (ValueError, KeyError):
                                continue

                        if dates and values:
                            series_data = pd.Series(values, index=dates, name=series_id)
                            data[series_id] = series_data
                            print(
                                f"✓ Retrieved {len(series_data)} observations for {series_id}"
                            )
                        else:
                            print(f"✗ No valid data for {series_id}")
                    else:
                        print(f"✗ No observations found for {series_id}")
                else:
                    print(f"✗ Error fetching {series_id}: HTTP {response.status_code}")

            except Exception as e:
                print(f"✗ Error fetching {series_id}: {e}")

        return data

    def create_dataframe(self, data_dict):
        """Convert dictionary of series data to a pandas DataFrame."""
        if not data_dict:
            return pd.DataFrame()

        # Find the common date range
        all_dates = set()
        for series in data_dict.values():
            all_dates.update(series.index)

        # Create a complete date range
        if all_dates:
            date_range = pd.date_range(min(all_dates), max(all_dates), freq="D")
            df = pd.DataFrame(index=date_range)

            # Add each series
            for series_id, series_data in data_dict.items():
                df[series_id] = series_data

            df.index.name = "Date"
            return df

        return pd.DataFrame()

    def save_data(self, df, filename):
        """Save data to CSV file."""
        if df.empty:
            print("No data to save")
            return None

        filepath = os.path.join(OUTPUT_DIR, filename)
        df.to_csv(filepath)
        print(f"Data saved to {filepath}")
        return filepath

    def plot_economic_indicators(self, df, indicators_to_plot=None):
        """Create plots for economic indicators."""
        if df.empty:
            print("No data to plot")
            return

        if indicators_to_plot is None:
            indicators_to_plot = [col for col in df.columns if col in df.columns]

        if not indicators_to_plot:
            print("No indicators to plot")
            return

        # Set up the plotting style
        plt.style.use("default")
        sns.set_palette("husl")

        # Create subplots
        n_indicators = len(indicators_to_plot)
        fig, axes = plt.subplots(n_indicators, 1, figsize=(15, 4 * n_indicators))

        if n_indicators == 1:
            axes = [axes]

        for i, indicator in enumerate(indicators_to_plot):
            if indicator in df.columns:
                ax = axes[i]
                df[indicator].dropna().plot(ax=ax, linewidth=2)

                # Get series info for title
                info = self.get_series_info(indicator)
                title = f'{indicator} - {info["title"]}' if info else indicator
                ax.set_title(title)
                ax.set_ylabel("Value")
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(PLOTS_DIR, "economic_indicators.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.show()
        print(f"Plot saved to {plot_path}")

    def generate_summary_statistics(self, df):
        """Generate summary statistics for the economic data."""
        if df.empty:
            return pd.DataFrame()

        summary = df.describe()

        # Add additional statistics
        summary.loc["missing_values"] = df.isnull().sum()
        summary.loc["missing_percentage"] = (df.isnull().sum() / len(df)) * 100

        return summary

    def run_analysis(self, series_ids=None, start_date=None, end_date=None):
        """Run a complete analysis of economic indicators."""
        if series_ids is None:
            series_ids = list(self.indicators.values())

        print("=== FRED Economic Data Analysis v2 ===")
        print(f"API Key: {self.api_key[:8]}...")
        print(
            f"Date Range: {start_date or DEFAULT_START_DATE} to {end_date or DEFAULT_END_DATE}"
        )
        print(f"Series to analyze: {series_ids}")
        print("=" * 50)

        # Fetch data
        data = self.get_economic_data(series_ids, start_date, end_date)

        if not data:
            print("No data retrieved. Please check your API key and series IDs.")
            return None, None

        # Create DataFrame
        df = self.create_dataframe(data)

        if df.empty:
            print("No data to analyze")
            return None, None

        # Save data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_data(df, f"fred_economic_data_{timestamp}.csv")

        # Generate summary statistics
        summary = self.generate_summary_statistics(df)
        print("\n=== Summary Statistics ===")
        print(summary)

        # Create plots
        print("\n=== Creating Visualizations ===")
        self.plot_economic_indicators(df)

        return df, summary


def main():
    """Main function to run the FRED data analysis."""
    collector = FREDDataCollectorV2()

    # Example: Analyze key economic indicators
    key_indicators = ["GDP", "UNRATE", "CPIAUCSL", "FEDFUNDS", "DGS10"]

    try:
        df, summary = collector.run_analysis(series_ids=key_indicators)

        if df is not None:
            print("\n=== Analysis Complete ===")
            print(f"Data shape: {df.shape}")
            print(f"Date range: {df.index.min()} to {df.index.max()}")
        else:
            print("\n=== Analysis Failed ===")

    except Exception as e:
        print(f"Error during analysis: {e}")


if __name__ == "__main__":
    main()
