#!/usr/bin/env python3
"""
Alignment and Divergence Analysis Test
Test the new alignment/divergence analyzer with real FRED data
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.core.enhanced_fred_client import EnhancedFREDClient
from src.analysis.alignment_divergence_analyzer import AlignmentDivergenceAnalyzer

def test_alignment_divergence_analysis():
    """Test the new alignment and divergence analysis"""
    
    # Use the provided API key
    api_key = "acf8bbec7efe3b6dfa6ae083e7152314"
    
    print("=== ALIGNMENT & DIVERGENCE ANALYSIS TEST ===")
    print("Using Spearman correlation for long-term alignment detection")
    print("Using Z-score analysis for sudden deviation detection")
    print()
    
    try:
        # Initialize FRED client
        client = EnhancedFREDClient(api_key)
        
        # Fetch economic data (last 5 years for better trend analysis)
        end_date = datetime.now()
        start_date = end_date.replace(year=end_date.year - 5)
        
        print("1. Fetching economic data...")
        data = client.fetch_economic_data(
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )
        
        if data.empty:
            print("❌ No data fetched")
            return
        
        print(f"✅ Fetched {len(data)} observations across {len(data.columns)} indicators")
        print(f"   Date range: {data.index.min()} to {data.index.max()}")
        print(f"   Indicators: {list(data.columns)}")
        print()
        
        # Initialize alignment analyzer
        analyzer = AlignmentDivergenceAnalyzer(data)
        
        # 2. Analyze long-term alignment using Spearman correlation
        print("2. Analyzing long-term alignment (Spearman correlation)...")
        alignment_results = analyzer.analyze_long_term_alignment(
            window_sizes=[12, 24, 48],  # 1, 2, 4 years for quarterly data
            min_periods=8
        )
        
        print("✅ Long-term alignment analysis completed")
        print(f"   Analyzed {len(alignment_results['rolling_correlations'])} indicator pairs")
        
        # Show alignment summary
        summary = alignment_results['alignment_summary']
        print(f"   Increasing alignment pairs: {len(summary['increasing_alignment'])}")
        print(f"   Decreasing alignment pairs: {len(summary['decreasing_alignment'])}")
        print(f"   Stable alignment pairs: {len(summary['stable_alignment'])}")
        print(f"   Strong trends: {len(summary['strong_trends'])}")
        print()
        
        # Show some specific alignment trends
        if summary['increasing_alignment']:
            print("🔺 Examples of increasing alignment:")
            for pair in summary['increasing_alignment'][:3]:
                print(f"   - {pair}")
            print()
        
        if summary['decreasing_alignment']:
            print("🔻 Examples of decreasing alignment:")
            for pair in summary['decreasing_alignment'][:3]:
                print(f"   - {pair}")
            print()
        
        # 3. Detect sudden deviations using Z-score analysis
        print("3. Detecting sudden deviations (Z-score analysis)...")
        deviation_results = analyzer.detect_sudden_deviations(
            z_threshold=2.0,  # Flag deviations beyond 2 standard deviations
            window_size=12,    # 3-year rolling window for quarterly data
            min_periods=6
        )
        
        print("✅ Sudden deviation detection completed")
        
        # Show deviation summary
        dev_summary = deviation_results['deviation_summary']
        print(f"   Total deviations detected: {dev_summary['total_deviations']}")
        print(f"   Indicators with deviations: {len(dev_summary['indicators_with_deviations'])}")
        print(f"   Extreme events: {dev_summary['extreme_events_count']}")
        print()
        
        # Show most volatile indicators
        if dev_summary['most_volatile_indicators']:
            print("📈 Most volatile indicators:")
            for item in dev_summary['most_volatile_indicators'][:5]:
                print(f"   - {item['indicator']}: {item['volatility']:.4f} volatility")
            print()
        
        # Show extreme events
        extreme_events = deviation_results['extreme_events']
        if extreme_events:
            print("🚨 Recent extreme events (Z-score > 3.0):")
            for indicator, events in extreme_events.items():
                if events['events']:
                    extreme_events_list = [e for e in events['events'] if abs(e['z_score']) > 3.0]
                    if extreme_events_list:
                        latest = extreme_events_list[0]
                        print(f"   - {indicator}: {latest['date'].strftime('%Y-%m-%d')} "
                              f"(Z-score: {latest['z_score']:.2f}, Growth: {latest['growth_rate']:.2f}%)")
            print()
        
        # 4. Generate insights report
        print("4. Generating comprehensive insights report...")
        insights_report = analyzer.generate_insights_report()
        print("✅ Insights report generated")
        print()
        
        # Save insights to file
        with open('alignment_divergence_insights.txt', 'w') as f:
            f.write(insights_report)
        print("📄 Insights report saved to 'alignment_divergence_insights.txt'")
        print()
        
        # 5. Create visualization
        print("5. Creating alignment analysis visualization...")
        analyzer.plot_alignment_analysis(save_path='alignment_analysis_plot.png')
        print("📊 Visualization saved to 'alignment_analysis_plot.png'")
        print()
        
        # 6. Detailed analysis examples
        print("6. Detailed analysis examples:")
        print()
        
        # Show specific correlation trends
        if alignment_results['trend_analysis']:
            print("📊 Correlation Trend Examples:")
            for pair_name, trends in list(alignment_results['trend_analysis'].items())[:3]:
                print(f"   {pair_name}:")
                for window_name, trend_info in trends.items():
                    if trend_info['trend'] != 'insufficient_data':
                        print(f"     {window_name}: {trend_info['trend']} ({trend_info['strength']})")
                        print(f"       Slope: {trend_info['slope']:.4f}, R²: {trend_info['r_squared']:.3f}")
                print()
        
        # Show specific deviation patterns
        if deviation_results['z_scores']:
            print("⚠️ Deviation Pattern Examples:")
            for indicator, z_scores in list(deviation_results['z_scores'].items())[:3]:
                deviations = deviation_results['deviations'][indicator]
                if not deviations.empty:
                    print(f"   {indicator}:")
                    print(f"     Total deviations: {len(deviations)}")
                    print(f"     Max Z-score: {deviations.abs().max():.2f}")
                    print(f"     Mean Z-score: {deviations.abs().mean():.2f}")
                    print(f"     Recent deviations: {len(deviations[deviations.index > '2023-01-01'])}")
                print()
        
        print("=== ANALYSIS COMPLETED SUCCESSFULLY ===")
        print("✅ Spearman correlation analysis for long-term alignment")
        print("✅ Z-score analysis for sudden deviation detection")
        print("✅ Comprehensive insights and visualizations generated")
        print()
        print("Key findings:")
        print("- Long-term alignment patterns identified using rolling Spearman correlation")
        print("- Sudden deviations flagged using Z-score analysis")
        print("- Extreme events detected and categorized")
        print("- Volatility patterns analyzed across indicators")
        
    except Exception as e:
        print(f"❌ Error during alignment/divergence analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_alignment_divergence_analysis() 