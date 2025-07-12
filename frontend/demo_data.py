"""
FRED ML - Demo Data Generator
Provides realistic economic data and senior data scientist insights
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_economic_data():
    """Generate realistic economic data for demonstration"""
    
    # Generate date range (last 5 years)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*5)
    dates = pd.date_range(start=start_date, end=end_date, freq='M')
    
    # Base values and trends for realistic economic data
    base_values = {
        'GDPC1': 20000,  # Real GDP in billions
        'INDPRO': 100,    # Industrial Production Index
        'RSAFS': 500,     # Retail Sales in billions
        'CPIAUCSL': 250,  # Consumer Price Index
        'FEDFUNDS': 2.5,  # Federal Funds Rate
        'DGS10': 3.0,     # 10-Year Treasury Rate
        'UNRATE': 4.0,    # Unemployment Rate
        'PAYEMS': 150000, # Total Nonfarm Payrolls (thousands)
        'PCE': 18000,     # Personal Consumption Expenditures
        'M2SL': 21000,    # M2 Money Stock
        'TCU': 75,        # Capacity Utilization
        'DEXUSEU': 1.1    # US/Euro Exchange Rate
    }
    
    # Growth rates and volatility for realistic trends
    growth_rates = {
        'GDPC1': 0.02,    # 2% annual growth
        'INDPRO': 0.015,  # 1.5% annual growth
        'RSAFS': 0.03,    # 3% annual growth
        'CPIAUCSL': 0.025, # 2.5% annual inflation
        'FEDFUNDS': 0.0,  # Policy rate
        'DGS10': 0.0,     # Market rate
        'UNRATE': 0.0,    # Unemployment
        'PAYEMS': 0.015,  # Employment growth
        'PCE': 0.025,     # Consumption growth
        'M2SL': 0.04,     # Money supply growth
        'TCU': 0.005,     # Capacity utilization
        'DEXUSEU': 0.0    # Exchange rate
    }
    
    # Generate realistic data
    data = {'Date': dates}
    
    for indicator, base_value in base_values.items():
        # Create trend with realistic economic cycles
        trend = np.linspace(0, len(dates) * growth_rates[indicator], len(dates))
        
        # Add business cycle effects
        cycle = 0.05 * np.sin(2 * np.pi * np.arange(len(dates)) / 48)  # 4-year cycle
        
        # Add random noise
        noise = np.random.normal(0, 0.02, len(dates))
        
        # Combine components
        values = base_value * (1 + trend + cycle + noise)
        
        # Ensure realistic bounds
        if indicator in ['UNRATE', 'FEDFUNDS', 'DGS10']:
            values = np.clip(values, 0, 20)
        elif indicator in ['CPIAUCSL']:
            values = np.clip(values, 200, 350)
        elif indicator in ['TCU']:
            values = np.clip(values, 60, 90)
        
        data[indicator] = values
    
    return pd.DataFrame(data)

def generate_insights():
    """Generate senior data scientist insights"""
    
    insights = {
        'GDPC1': {
            'current_value': '$21,847.2B',
            'growth_rate': '+2.1%',
            'trend': 'Moderate growth',
            'forecast': '+2.3% next quarter',
            'key_insight': 'GDP growth remains resilient despite monetary tightening, supported by strong consumer spending and business investment.',
            'risk_factors': ['Inflation persistence', 'Geopolitical tensions', 'Supply chain disruptions'],
            'opportunities': ['Technology sector expansion', 'Infrastructure investment', 'Green energy transition']
        },
        'INDPRO': {
            'current_value': '102.4',
            'growth_rate': '+0.8%',
            'trend': 'Recovery phase',
            'forecast': '+0.6% next month',
            'key_insight': 'Industrial production shows signs of recovery, with manufacturing leading the rebound. Capacity utilization improving.',
            'risk_factors': ['Supply chain bottlenecks', 'Labor shortages', 'Energy price volatility'],
            'opportunities': ['Advanced manufacturing', 'Automation adoption', 'Reshoring initiatives']
        },
        'RSAFS': {
            'current_value': '$579.2B',
            'growth_rate': '+3.2%',
            'trend': 'Strong consumer spending',
            'forecast': '+2.8% next month',
            'key_insight': 'Retail sales demonstrate robust consumer confidence, with e-commerce continuing to gain market share.',
            'risk_factors': ['Inflation impact on purchasing power', 'Interest rate sensitivity', 'Supply chain issues'],
            'opportunities': ['Digital transformation', 'Omnichannel retail', 'Personalization']
        },
        'CPIAUCSL': {
            'current_value': '312.3',
            'growth_rate': '+3.2%',
            'trend': 'Moderating inflation',
            'forecast': '+2.9% next month',
            'key_insight': 'Inflation continues to moderate from peak levels, with core CPI showing signs of stabilization.',
            'risk_factors': ['Energy price volatility', 'Wage pressure', 'Supply chain costs'],
            'opportunities': ['Productivity improvements', 'Technology adoption', 'Supply chain optimization']
        },
        'FEDFUNDS': {
            'current_value': '5.25%',
            'growth_rate': '0%',
            'trend': 'Stable policy rate',
            'forecast': '5.25% next meeting',
            'key_insight': 'Federal Reserve maintains restrictive stance to combat inflation, with policy rate at 22-year high.',
            'risk_factors': ['Inflation persistence', 'Economic slowdown', 'Financial stability'],
            'opportunities': ['Policy normalization', 'Inflation targeting', 'Financial regulation']
        },
        'DGS10': {
            'current_value': '4.12%',
            'growth_rate': '-0.15%',
            'trend': 'Declining yields',
            'forecast': '4.05% next week',
            'key_insight': '10-year Treasury yields declining on economic uncertainty and flight to quality. Yield curve inversion persists.',
            'risk_factors': ['Economic recession', 'Inflation expectations', 'Geopolitical risks'],
            'opportunities': ['Bond market opportunities', 'Portfolio diversification', 'Interest rate hedging']
        },
        'UNRATE': {
            'current_value': '3.7%',
            'growth_rate': '0%',
            'trend': 'Stable employment',
            'forecast': '3.6% next month',
            'key_insight': 'Unemployment rate remains near historic lows, indicating tight labor market conditions.',
            'risk_factors': ['Labor force participation', 'Skills mismatch', 'Economic slowdown'],
            'opportunities': ['Workforce development', 'Technology training', 'Remote work adoption']
        },
        'PAYEMS': {
            'current_value': '156,847K',
            'growth_rate': '+1.2%',
            'trend': 'Steady job growth',
            'forecast': '+0.8% next month',
            'key_insight': 'Nonfarm payrolls continue steady growth, with healthcare and technology sectors leading job creation.',
            'risk_factors': ['Labor shortages', 'Wage pressure', 'Economic uncertainty'],
            'opportunities': ['Skills development', 'Industry partnerships', 'Immigration policy']
        },
        'PCE': {
            'current_value': '$19,847B',
            'growth_rate': '+2.8%',
            'trend': 'Strong consumption',
            'forecast': '+2.5% next quarter',
            'key_insight': 'Personal consumption expenditures show resilience, supported by strong labor market and wage growth.',
            'risk_factors': ['Inflation impact', 'Interest rate sensitivity', 'Consumer confidence'],
            'opportunities': ['Digital commerce', 'Experience economy', 'Sustainable consumption']
        },
        'M2SL': {
            'current_value': '$20,847B',
            'growth_rate': '+2.1%',
            'trend': 'Moderate growth',
            'forecast': '+1.8% next month',
            'key_insight': 'Money supply growth moderating as Federal Reserve tightens monetary policy to combat inflation.',
            'risk_factors': ['Inflation expectations', 'Financial stability', 'Economic growth'],
            'opportunities': ['Digital payments', 'Financial innovation', 'Monetary policy']
        },
        'TCU': {
            'current_value': '78.4%',
            'growth_rate': '+0.3%',
            'trend': 'Improving utilization',
            'forecast': '78.7% next quarter',
            'key_insight': 'Capacity utilization improving as supply chain issues resolve and demand remains strong.',
            'risk_factors': ['Supply chain disruptions', 'Labor shortages', 'Energy constraints'],
            'opportunities': ['Efficiency improvements', 'Technology adoption', 'Process optimization']
        },
        'DEXUSEU': {
            'current_value': '1.087',
            'growth_rate': '+0.2%',
            'trend': 'Stable exchange rate',
            'forecast': '1.085 next week',
            'key_insight': 'US dollar remains strong against euro, supported by relative economic performance and interest rate differentials.',
            'risk_factors': ['Economic divergence', 'Geopolitical tensions', 'Trade policies'],
            'opportunities': ['Currency hedging', 'International trade', 'Investment diversification']
        }
    }
    
    return insights

def generate_forecast_data():
    """Generate forecast data with confidence intervals"""
    
    # Generate future dates (next 4 quarters)
    last_date = datetime.now()
    future_dates = pd.date_range(start=last_date + timedelta(days=90), periods=4, freq='Q')
    
    forecasts = {}
    
    # Realistic forecast scenarios
    forecast_scenarios = {
        'GDPC1': {'growth': 0.02, 'volatility': 0.01},  # 2% quarterly growth
        'INDPRO': {'growth': 0.015, 'volatility': 0.008},  # 1.5% monthly growth
        'RSAFS': {'growth': 0.025, 'volatility': 0.012},  # 2.5% monthly growth
        'CPIAUCSL': {'growth': 0.006, 'volatility': 0.003},  # 0.6% monthly inflation
        'FEDFUNDS': {'growth': 0.0, 'volatility': 0.25},  # Stable policy rate
        'DGS10': {'growth': -0.001, 'volatility': 0.15},  # Slight decline
        'UNRATE': {'growth': -0.001, 'volatility': 0.1},  # Slight decline
        'PAYEMS': {'growth': 0.008, 'volatility': 0.005},  # 0.8% monthly growth
        'PCE': {'growth': 0.02, 'volatility': 0.01},  # 2% quarterly growth
        'M2SL': {'growth': 0.015, 'volatility': 0.008},  # 1.5% monthly growth
        'TCU': {'growth': 0.003, 'volatility': 0.002},  # 0.3% quarterly growth
        'DEXUSEU': {'growth': -0.001, 'volatility': 0.02}  # Slight decline
    }
    
    for indicator, scenario in forecast_scenarios.items():
        base_value = 100  # Normalized base value
        
        # Generate forecast values
        forecast_values = []
        confidence_intervals = []
        
        for i in range(4):
            # Add trend and noise
            value = base_value * (1 + scenario['growth'] * (i + 1) + 
                                np.random.normal(0, scenario['volatility']))
            
            # Generate confidence interval
            lower = value * (1 - 0.05 - np.random.uniform(0, 0.03))
            upper = value * (1 + 0.05 + np.random.uniform(0, 0.03))
            
            forecast_values.append(value)
            confidence_intervals.append({'lower': lower, 'upper': upper})
        
        forecasts[indicator] = {
            'forecast': forecast_values,
            'confidence_intervals': pd.DataFrame(confidence_intervals),
            'dates': future_dates
        }
    
    return forecasts

def generate_correlation_matrix():
    """Generate realistic correlation matrix"""
    
    # Define realistic correlations between economic indicators
    correlations = {
        'GDPC1': {'INDPRO': 0.85, 'RSAFS': 0.78, 'CPIAUCSL': 0.45, 'FEDFUNDS': -0.32, 'DGS10': -0.28},
        'INDPRO': {'RSAFS': 0.72, 'CPIAUCSL': 0.38, 'FEDFUNDS': -0.25, 'DGS10': -0.22},
        'RSAFS': {'CPIAUCSL': 0.42, 'FEDFUNDS': -0.28, 'DGS10': -0.25},
        'CPIAUCSL': {'FEDFUNDS': 0.65, 'DGS10': 0.58},
        'FEDFUNDS': {'DGS10': 0.82}
    }
    
    # Create correlation matrix
    indicators = ['GDPC1', 'INDPRO', 'RSAFS', 'CPIAUCSL', 'FEDFUNDS', 'DGS10', 'UNRATE', 'PAYEMS', 'PCE', 'M2SL', 'TCU', 'DEXUSEU']
    corr_matrix = pd.DataFrame(index=indicators, columns=indicators)
    
    # Fill diagonal with 1
    for indicator in indicators:
        corr_matrix.loc[indicator, indicator] = 1.0
    
    # Fill with realistic correlations
    for i, indicator1 in enumerate(indicators):
        for j, indicator2 in enumerate(indicators):
            if i != j:
                if indicator1 in correlations and indicator2 in correlations[indicator1]:
                    corr_matrix.loc[indicator1, indicator2] = correlations[indicator1][indicator2]
                elif indicator2 in correlations and indicator1 in correlations[indicator2]:
                    corr_matrix.loc[indicator1, indicator2] = correlations[indicator2][indicator1]
                else:
                    # Generate random correlation between -0.3 and 0.3
                    corr_matrix.loc[indicator1, indicator2] = np.random.uniform(-0.3, 0.3)
    
    return corr_matrix

def get_demo_data():
    """Get comprehensive demo data"""
    return {
        'economic_data': generate_economic_data(),
        'insights': generate_insights(),
        'forecasts': generate_forecast_data(),
        'correlation_matrix': generate_correlation_matrix()
    } 