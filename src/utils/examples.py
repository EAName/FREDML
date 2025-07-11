#!/usr/bin/env python3
"""
Example usage of the FRED Data Collector
Demonstrates various ways to use the tool for economic data analysis
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime, timedelta

import pandas as pd

from core.fred_client import FREDDataCollectorV2


def example_basic_usage():
    """Basic usage example."""
    print("=== Basic Usage Example ===")

    collector = FREDDataCollectorV2()

    # Get data for a single indicator
    gdp_data = collector.get_economic_data(["GDP"], "2020-01-01", "2024-01-01")
    df = collector.create_dataframe(gdp_data)

    print(f"GDP data shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Latest GDP value: ${df['GDP'].iloc[-1]:,.2f} billion")

    return df


def example_multiple_indicators():
    """Example with multiple economic indicators."""
    print("\n=== Multiple Indicators Example ===")

    collector = FREDDataCollectorV2()

    # Define indicators of interest
    indicators = ["UNRATE", "CPIAUCSL", "FEDFUNDS"]

    # Get data for the last 5 years
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=5 * 365)).strftime("%Y-%m-%d")

    data = collector.get_economic_data(indicators, start_date, end_date)
    df = collector.create_dataframe(data)

    # Generate summary statistics
    summary = collector.generate_summary_statistics(df)
    print("\nSummary Statistics:")
    print(summary)

    # Save data
    collector.save_data(df, "example_multiple_indicators.csv")

    return df


def example_custom_analysis():
    """Example of custom analysis."""
    print("\n=== Custom Analysis Example ===")

    collector = FREDDataCollectorV2()

    # Focus on monetary policy indicators
    monetary_indicators = ["FEDFUNDS", "DGS10", "M2SL"]

    # Get data for the last 10 years
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=10 * 365)).strftime("%Y-%m-%d")

    data = collector.get_economic_data(monetary_indicators, start_date, end_date)
    df = collector.create_dataframe(data)

    # Calculate some custom metrics
    if "FEDFUNDS" in df.columns and "DGS10" in df.columns:
        # Calculate yield curve spread (10Y - Fed Funds)
        df["YIELD_SPREAD"] = df["DGS10"] - df["FEDFUNDS"]

        print(f"\nYield Curve Analysis:")
        print(f"Current Fed Funds Rate: {df['FEDFUNDS'].iloc[-1]:.2f}%")
        print(f"Current 10Y Treasury Rate: {df['DGS10'].iloc[-1]:.2f}%")
        print(f"Current Yield Spread: {df['YIELD_SPREAD'].iloc[-1]:.2f}%")

        # Check for inverted yield curve (negative spread)
        inverted_periods = df[df["YIELD_SPREAD"] < 0]
        if not inverted_periods.empty:
            print(f"Yield curve inverted for {len(inverted_periods)} periods")

    return df


def example_series_info():
    """Example of getting series information."""
    print("\n=== Series Information Example ===")

    collector = FREDDataCollectorV2()

    # Get information about different series
    series_to_check = ["GDP", "UNRATE", "CPIAUCSL"]

    for series_id in series_to_check:
        info = collector.get_series_info(series_id)
        if info:
            print(f"\n{series_id}:")
            print(f"  Title: {info['title']}")
            print(f"  Units: {info['units']}")
            print(f"  Frequency: {info['frequency']}")
            print(f"  Last Updated: {info['last_updated']}")


def example_error_handling():
    """Example showing error handling."""
    print("\n=== Error Handling Example ===")

    collector = FREDDataCollectorV2()

    # Try to get data for an invalid series ID
    invalid_series = ["INVALID_SERIES_ID"]

    data = collector.get_economic_data(invalid_series)
    print("Attempted to fetch invalid series - handled gracefully")


def main():
    """Run all examples."""
    print("FRED Data Collector - Example Usage")
    print("=" * 50)

    try:
        # Run examples
        example_basic_usage()
        example_multiple_indicators()
        example_custom_analysis()
        example_series_info()
        example_error_handling()

        print("\n=== All Examples Completed Successfully ===")

    except Exception as e:
        print(f"Error running examples: {e}")


if __name__ == "__main__":
    main()
