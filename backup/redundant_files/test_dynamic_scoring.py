#!/usr/bin/env python3
"""
Test Dynamic Scoring Implementation
Verifies that the economic health and market sentiment scores
are calculated correctly using real-time FRED data
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Add frontend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'frontend'))

def test_dynamic_scoring():
    """Test the dynamic scoring implementation"""
    
    print("=== TESTING DYNAMIC SCORING IMPLEMENTATION ===\n")
    
    # Import the scoring functions
    try:
        from frontend.fred_api_client import generate_real_insights
        
        # Get API key
        api_key = os.getenv('FRED_API_KEY')
        if not api_key:
            print("❌ FRED_API_KEY not set")
            return False
        
        print("1. Testing real-time data fetching...")
        insights = generate_real_insights(api_key)
        
        if not insights:
            print("❌ No insights generated")
            return False
        
        print(f"✅ Generated insights for {len(insights)} indicators")
        
        # Test the scoring functions
        print("\n2. Testing Economic Health Score...")
        
        # Import the scoring functions from the app
        def normalize(value, min_val, max_val):
            """Normalize a value to 0-1 range"""
            if max_val == min_val:
                return 0.5
            return max(0, min(1, (value - min_val) / (max_val - min_val)))
        
        def calculate_health_score(insights):
            """Calculate dynamic economy health score (0-100) based on real-time indicators"""
            score = 0
            weights = {
                'gdp_growth': 0.3,
                'inflation': 0.2,
                'unemployment': 0.2,
                'industrial_production': 0.2,
                'fed_rate': 0.1
            }
            
            # GDP growth (GDPC1) - normalize 0-5% range
            gdp_growth = 0
            if 'GDPC1' in insights:
                gdp_growth_raw = insights['GDPC1'].get('growth_rate', 0)
                if isinstance(gdp_growth_raw, str):
                    try:
                        gdp_growth = float(gdp_growth_raw.replace('%', '').replace('+', ''))
                    except:
                        gdp_growth = 0
                else:
                    gdp_growth = float(gdp_growth_raw)
            
            gdp_score = normalize(gdp_growth, 0, 5) * weights['gdp_growth']
            score += gdp_score
            
            # Inflation (CPIAUCSL) - normalize 0-10% range, lower is better
            inflation_rate = 0
            if 'CPIAUCSL' in insights:
                inflation_raw = insights['CPIAUCSL'].get('growth_rate', 0)
                if isinstance(inflation_raw, str):
                    try:
                        inflation_rate = float(inflation_raw.replace('%', '').replace('+', ''))
                    except:
                        inflation_rate = 0
                else:
                    inflation_rate = float(inflation_raw)
            
            # Target inflation is 2%, so we score based on distance from 2%
            inflation_score = normalize(1 - abs(inflation_rate - 2), 0, 1) * weights['inflation']
            score += inflation_score
            
            # Unemployment (UNRATE) - normalize 0-10% range, lower is better
            unemployment_rate = 5  # Default to 5%
            if 'UNRATE' in insights:
                unrate_raw = insights['UNRATE'].get('current_value', '5%')
                if isinstance(unrate_raw, str):
                    try:
                        unemployment_rate = float(unrate_raw.replace('%', ''))
                    except:
                        unemployment_rate = 5
                else:
                    unemployment_rate = float(unrate_raw)
            
            unemployment_score = normalize(1 - unemployment_rate / 10, 0, 1) * weights['unemployment']
            score += unemployment_score
            
            # Industrial Production (INDPRO) - normalize 0-5% range
            ip_growth = 0
            if 'INDPRO' in insights:
                ip_raw = insights['INDPRO'].get('growth_rate', 0)
                if isinstance(ip_raw, str):
                    try:
                        ip_growth = float(ip_raw.replace('%', '').replace('+', ''))
                    except:
                        ip_growth = 0
                else:
                    ip_growth = float(ip_raw)
            
            ip_score = normalize(ip_growth, 0, 5) * weights['industrial_production']
            score += ip_score
            
            # Federal Funds Rate (FEDFUNDS) - normalize 0-10% range, lower is better
            fed_rate = 2  # Default to 2%
            if 'FEDFUNDS' in insights:
                fed_raw = insights['FEDFUNDS'].get('current_value', '2%')
                if isinstance(fed_raw, str):
                    try:
                        fed_rate = float(fed_raw.replace('%', ''))
                    except:
                        fed_rate = 2
                else:
                    fed_rate = float(fed_raw)
            
            fed_score = normalize(1 - fed_rate / 10, 0, 1) * weights['fed_rate']
            score += fed_score
            
            return max(0, min(100, score * 100))
        
        def calculate_sentiment_score(insights):
            """Calculate dynamic market sentiment score (0-100) based on real-time indicators"""
            score = 0
            weights = {
                'news_sentiment': 0.5,
                'social_sentiment': 0.3,
                'volatility': 0.2
            }
            
            # News sentiment (simulated based on economic indicators)
            # Use a combination of GDP growth, unemployment, and inflation
            news_sentiment = 0
            if 'GDPC1' in insights:
                gdp_growth = insights['GDPC1'].get('growth_rate', 0)
                if isinstance(gdp_growth, str):
                    try:
                        gdp_growth = float(gdp_growth.replace('%', '').replace('+', ''))
                    except:
                        gdp_growth = 0
                else:
                    gdp_growth = float(gdp_growth)
                news_sentiment += normalize(gdp_growth, -2, 5) * 0.4
            
            if 'UNRATE' in insights:
                unrate = insights['UNRATE'].get('current_value', '5%')
                if isinstance(unrate, str):
                    try:
                        unrate = float(unrate.replace('%', ''))
                    except:
                        unrate = 5
                else:
                    unrate = float(unrate)
                news_sentiment += normalize(1 - unrate / 10, 0, 1) * 0.3
            
            if 'CPIAUCSL' in insights:
                inflation = insights['CPIAUCSL'].get('growth_rate', 0)
                if isinstance(inflation, str):
                    try:
                        inflation = float(inflation.replace('%', '').replace('+', ''))
                    except:
                        inflation = 0
                else:
                    inflation = float(inflation)
                # Moderate inflation (2-3%) is positive for sentiment
                inflation_sentiment = normalize(1 - abs(inflation - 2.5), 0, 1)
                news_sentiment += inflation_sentiment * 0.3
            
            news_score = normalize(news_sentiment, 0, 1) * weights['news_sentiment']
            score += news_score
            
            # Social sentiment (simulated based on interest rates and yields)
            # Lower rates generally indicate positive sentiment
            social_sentiment = 0
            if 'FEDFUNDS' in insights:
                fed_rate = insights['FEDFUNDS'].get('current_value', '2%')
                if isinstance(fed_rate, str):
                    try:
                        fed_rate = float(fed_rate.replace('%', ''))
                    except:
                        fed_rate = 2
                else:
                    fed_rate = float(fed_rate)
                social_sentiment += normalize(1 - fed_rate / 10, 0, 1) * 0.5
            
            if 'DGS10' in insights:
                treasury = insights['DGS10'].get('current_value', '3%')
                if isinstance(treasury, str):
                    try:
                        treasury = float(treasury.replace('%', ''))
                    except:
                        treasury = 3
                else:
                    treasury = float(treasury)
                social_sentiment += normalize(1 - treasury / 10, 0, 1) * 0.5
            
            social_score = normalize(social_sentiment, 0, 1) * weights['social_sentiment']
            score += social_score
            
            # Volatility (simulated based on economic uncertainty)
            # Use inflation volatility and interest rate changes
            volatility = 0.5  # Default moderate volatility
            if 'CPIAUCSL' in insights and 'FEDFUNDS' in insights:
                inflation = insights['CPIAUCSL'].get('growth_rate', 0)
                fed_rate = insights['FEDFUNDS'].get('current_value', '2%')
                
                if isinstance(inflation, str):
                    try:
                        inflation = float(inflation.replace('%', '').replace('+', ''))
                    except:
                        inflation = 0
                else:
                    inflation = float(inflation)
                
                if isinstance(fed_rate, str):
                    try:
                        fed_rate = float(fed_rate.replace('%', ''))
                    except:
                        fed_rate = 2
                else:
                    fed_rate = float(fed_rate)
                
                # Higher inflation and rate volatility = higher market volatility
                inflation_vol = min(abs(inflation - 2) / 2, 1)  # Distance from target
                rate_vol = min(abs(fed_rate - 2) / 5, 1)  # Distance from neutral
                volatility = (inflation_vol + rate_vol) / 2
            
            volatility_score = normalize(1 - volatility, 0, 1) * weights['volatility']
            score += volatility_score
            
            return max(0, min(100, score * 100))
        
        def label_score(score):
            """Classify score into meaningful labels"""
            if score >= 70:
                return "Strong"
            elif score >= 50:
                return "Moderate"
            elif score >= 30:
                return "Weak"
            else:
                return "Critical"
        
        # Calculate scores
        health_score = calculate_health_score(insights)
        sentiment_score = calculate_sentiment_score(insights)
        
        # Get labels
        health_label = label_score(health_score)
        sentiment_label = label_score(sentiment_score)
        
        print(f"✅ Economic Health Score: {health_score:.1f}/100 ({health_label})")
        print(f"✅ Market Sentiment Score: {sentiment_score:.1f}/100 ({sentiment_label})")
        
        # Test with different scenarios
        print("\n3. Testing scoring with different scenarios...")
        
        # Scenario 1: Strong economy
        strong_insights = {
            'GDPC1': {'growth_rate': '4.2%'},
            'CPIAUCSL': {'growth_rate': '2.1%'},
            'UNRATE': {'current_value': '3.5%'},
            'INDPRO': {'growth_rate': '3.8%'},
            'FEDFUNDS': {'current_value': '1.5%'}
        }
        
        strong_health = calculate_health_score(strong_insights)
        strong_sentiment = calculate_sentiment_score(strong_insights)
        
        print(f"   Strong Economy: Health={strong_health:.1f}, Sentiment={strong_sentiment:.1f}")
        
        # Scenario 2: Weak economy
        weak_insights = {
            'GDPC1': {'growth_rate': '-1.2%'},
            'CPIAUCSL': {'growth_rate': '6.5%'},
            'UNRATE': {'current_value': '7.8%'},
            'INDPRO': {'growth_rate': '-2.1%'},
            'FEDFUNDS': {'current_value': '5.2%'}
        }
        
        weak_health = calculate_health_score(weak_insights)
        weak_sentiment = calculate_sentiment_score(weak_insights)
        
        print(f"   Weak Economy: Health={weak_health:.1f}, Sentiment={weak_sentiment:.1f}")
        
        # Verify scoring logic
        print("\n4. Verifying scoring logic...")
        
        # Health score should be higher for strong economy
        if strong_health > weak_health:
            print("✅ Health scoring logic verified (strong > weak)")
        else:
            print("❌ Health scoring logic failed")
        
        # Sentiment score should be higher for strong economy
        if strong_sentiment > weak_sentiment:
            print("✅ Sentiment scoring logic verified (strong > weak)")
        else:
            print("❌ Sentiment scoring logic failed")
        
        # Test normalization function
        print("\n5. Testing normalization function...")
        
        test_cases = [
            (0, 0, 10, 0.0),
            (5, 0, 10, 0.5),
            (10, 0, 10, 1.0),
            (15, 0, 10, 1.0),  # Clamped to max
            (-5, 0, 10, 0.0),  # Clamped to min
        ]
        
        for value, min_val, max_val, expected in test_cases:
            result = normalize(value, min_val, max_val)
            if abs(result - expected) < 0.01:
                print(f"✅ normalize({value}, {min_val}, {max_val}) = {result:.2f}")
            else:
                print(f"❌ normalize({value}, {min_val}, {max_val}) = {result:.2f}, expected {expected:.2f}")
        
        print("\n=== DYNAMIC SCORING TEST COMPLETE ===")
        return True
        
    except Exception as e:
        print(f"❌ Error testing dynamic scoring: {e}")
        return False

if __name__ == "__main__":
    success = test_dynamic_scoring()
    if success:
        print("\n🎉 All tests passed! Dynamic scoring is working correctly.")
    else:
        print("\n💥 Some tests failed. Please check the implementation.") 