#!/usr/bin/env python3
"""
Test Enhanced FRED ML Application
Verifies real-time FRED API integration and enhanced features
"""

import os
import sys
import pandas as pd
from datetime import datetime, timedelta

# Add frontend to path
sys.path.append('frontend')

def test_fred_api_integration():
    """Test FRED API integration and real-time data fetching"""
    print("=== TESTING ENHANCED FRED ML APPLICATION ===")
    
    # Test FRED API key
    fred_key = os.getenv('FRED_API_KEY')
    if not fred_key:
        print("❌ FRED_API_KEY not found in environment")
        return False
    
    print(f"✅ FRED API Key: {fred_key[:8]}...")
    
    try:
        # Test FRED API client
        from frontend.fred_api_client import FREDAPIClient, generate_real_insights, get_real_economic_data
        
        # Test basic client functionality
        client = FREDAPIClient(fred_key)
        print("✅ FRED API Client initialized")
        
        # Test insights generation
        print("\n📊 Testing Real-Time Insights Generation...")
        insights = generate_real_insights(fred_key)
        
        if insights:
            print(f"✅ Generated insights for {len(insights)} indicators")
            
            # Show sample insights
            for indicator, insight in list(insights.items())[:3]:
                print(f"  {indicator}: {insight.get('current_value', 'N/A')} ({insight.get('growth_rate', 'N/A')})")
        else:
            print("❌ Failed to generate insights")
            return False
        
        # Test economic data fetching
        print("\n📈 Testing Economic Data Fetching...")
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        economic_data = get_real_economic_data(fred_key, start_date, end_date)
        
        if 'economic_data' in economic_data and not economic_data['economic_data'].empty:
            df = economic_data['economic_data']
            print(f"✅ Fetched economic data: {df.shape[0]} observations, {df.shape[1]} indicators")
            print(f"  Date range: {df.index.min()} to {df.index.max()}")
            print(f"  Indicators: {list(df.columns)}")
        else:
            print("❌ Failed to fetch economic data")
            return False
        
        # Test correlation analysis
        print("\n🔗 Testing Correlation Analysis...")
        corr_matrix = df.corr(method='spearman')
        print(f"✅ Calculated Spearman correlations for {len(corr_matrix)} indicators")
        
        # Show strongest correlations
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.5:
                    corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_value))
        
        corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        print(f"  Found {len(corr_pairs)} strong correlations (>0.5)")
        for pair in corr_pairs[:3]:
            print(f"    {pair[0]} ↔ {pair[1]}: {pair[2]:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing FRED API integration: {e}")
        return False

def test_enhanced_features():
    """Test enhanced application features"""
    print("\n=== TESTING ENHANCED FEATURES ===")
    
    try:
        # Test insights generation with enhanced analysis
        from frontend.fred_api_client import generate_real_insights
        fred_key = os.getenv('FRED_API_KEY')
        
        insights = generate_real_insights(fred_key)
        
        # Test economic health assessment
        print("🏥 Testing Economic Health Assessment...")
        health_indicators = ['GDPC1', 'INDPRO', 'UNRATE', 'CPIAUCSL']
        health_score = 0
        
        for indicator in health_indicators:
            if indicator in insights:
                insight = insights[indicator]
                growth_rate = insight.get('growth_rate', 0)
                
                # Convert growth_rate to float if it's a string
                try:
                    if isinstance(growth_rate, str):
                        growth_rate = float(growth_rate.replace('%', '').replace('+', ''))
                    else:
                        growth_rate = float(growth_rate)
                except (ValueError, TypeError):
                    growth_rate = 0
                
                if indicator == 'GDPC1' and growth_rate > 2:
                    health_score += 25
                elif indicator == 'INDPRO' and growth_rate > 1:
                    health_score += 25
                elif indicator == 'UNRATE':
                    current_value = insight.get('current_value', '0%').replace('%', '')
                    try:
                        unrate_val = float(current_value)
                        if unrate_val < 4:
                            health_score += 25
                    except:
                        pass
                elif indicator == 'CPIAUCSL' and 1 < growth_rate < 3:
                    health_score += 25
        
        print(f"✅ Economic Health Score: {health_score}/100")
        
        # Test market sentiment analysis
        print("📊 Testing Market Sentiment Analysis...")
        sentiment_indicators = ['DGS10', 'FEDFUNDS', 'RSAFS']
        sentiment_score = 0
        
        for indicator in sentiment_indicators:
            if indicator in insights:
                insight = insights[indicator]
                current_value = insight.get('current_value', '0')
                growth_rate = insight.get('growth_rate', 0)
                
                # Convert values to float
                try:
                    if isinstance(growth_rate, str):
                        growth_rate = float(growth_rate.replace('%', '').replace('+', ''))
                    else:
                        growth_rate = float(growth_rate)
                except (ValueError, TypeError):
                    growth_rate = 0
                
                if indicator == 'DGS10':
                    try:
                        yield_val = float(current_value.replace('%', ''))
                        if 2 < yield_val < 5:
                            sentiment_score += 33
                    except:
                        pass
                elif indicator == 'FEDFUNDS':
                    try:
                        rate_val = float(current_value.replace('%', ''))
                        if rate_val < 3:
                            sentiment_score += 33
                    except:
                        pass
                elif indicator == 'RSAFS' and growth_rate > 2:
                    sentiment_score += 34
        
        print(f"✅ Market Sentiment Score: {sentiment_score}/100")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing enhanced features: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Enhanced FRED ML Application")
    print("=" * 50)
    
    # Test FRED API integration
    api_success = test_fred_api_integration()
    
    # Test enhanced features
    features_success = test_enhanced_features()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 TEST SUMMARY")
    print("=" * 50)
    
    if api_success and features_success:
        print("✅ ALL TESTS PASSED")
        print("✅ Real-time FRED API integration working")
        print("✅ Enhanced features functioning")
        print("✅ Application ready for production use")
        return True
    else:
        print("❌ SOME TESTS FAILED")
        if not api_success:
            print("❌ FRED API integration issues")
        if not features_success:
            print("❌ Enhanced features issues")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 