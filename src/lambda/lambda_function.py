#!/usr/bin/env python3
"""
FRED ML Lambda Function
AWS Lambda function for processing economic data analysis
"""

import json
import os
import boto3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from datetime import datetime, timedelta
import requests
from typing import Dict, List, Optional, Tuple
import logging

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
s3_client = boto3.client('s3')
lambda_client = boto3.client('lambda')

# Configuration
FRED_API_KEY = os.environ.get('FRED_API_KEY')
S3_BUCKET = os.environ.get('S3_BUCKET', 'fredmlv1')
FRED_BASE_URL = "https://api.stlouisfed.org/fred"

# Economic indicators mapping
ECONOMIC_INDICATORS = {
    "GDP": "GDP",
    "UNRATE": "UNRATE", 
    "CPIAUCSL": "CPIAUCSL",
    "FEDFUNDS": "FEDFUNDS",
    "DGS10": "DGS10",
    "DEXUSEU": "DEXUSEU",
    "PAYEMS": "PAYEMS",
    "INDPRO": "INDPRO",
    "M2SL": "M2SL",
    "PCE": "PCE"
}

def get_fred_data(series_id: str, start_date: str, end_date: str) -> Optional[pd.Series]:
    """Fetch data from FRED API"""
    try:
        url = f"{FRED_BASE_URL}/series/observations"
        params = {
            "series_id": series_id,
            "api_key": FRED_API_KEY,
            "file_type": "json",
            "start_date": start_date,
            "end_date": end_date,
        }

        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            observations = data.get("observations", [])
            
            if observations:
                dates = []
                values = []
                
                for obs in observations:
                    try:
                        date = pd.to_datetime(obs["date"])
                        value = float(obs["value"]) if obs["value"] != "." else np.nan
                        dates.append(date)
                        values.append(value)
                    except (ValueError, KeyError):
                        continue
                
                if dates and values:
                    return pd.Series(values, index=dates, name=series_id)
        
        logger.error(f"Failed to fetch data for {series_id}")
        return None
        
    except Exception as e:
        logger.error(f"Error fetching data for {series_id}: {e}")
        return None

def create_dataframe(series_data: Dict[str, pd.Series]) -> pd.DataFrame:
    """Create DataFrame from series data"""
    if not series_data:
        return pd.DataFrame()
    
    # Find common date range
    all_dates = set()
    for series in series_data.values():
        if series is not None:
            all_dates.update(series.index)
    
    if all_dates:
        date_range = pd.date_range(min(all_dates), max(all_dates), freq='D')
        df = pd.DataFrame(index=date_range)
        
        for series_id, series_data in series_data.items():
            if series_data is not None:
                df[series_id] = series_data
        
        df.index.name = 'Date'
        return df
    
    return pd.DataFrame()

def generate_statistics(df: pd.DataFrame) -> Dict:
    """Generate statistical summary"""
    if df.empty:
        return {}
    
    stats = {}
    for column in df.columns:
        if column != 'Date':
            series = df[column].dropna()
            if not series.empty:
                stats[column] = {
                    'mean': float(series.mean()),
                    'std': float(series.std()),
                    'min': float(series.min()),
                    'max': float(series.max()),
                    'count': int(len(series)),
                    'missing': int(df[column].isna().sum())
                }
    
    return stats

def create_correlation_matrix(df: pd.DataFrame) -> Dict:
    """Create correlation matrix"""
    if df.empty:
        return {}
    
    corr_matrix = df.corr()
    return corr_matrix.to_dict()

def create_visualizations(df: pd.DataFrame, s3_bucket: str, report_id: str) -> List[str]:
    """Create and upload visualizations to S3"""
    if df.empty:
        return []
    
    visualization_keys = []
    
    try:
        # Time series plot
        plt.figure(figsize=(12, 8))
        for column in df.columns:
            if column != 'Date':
                plt.plot(df.index, df[column], label=column, linewidth=2)
        
        plt.title('Economic Indicators Time Series')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save to S3
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        
        time_series_key = f"visualizations/{report_id}/time_series.png"
        s3_client.put_object(
            Bucket=s3_bucket,
            Key=time_series_key,
            Body=img_buffer.getvalue(),
            ContentType='image/png'
        )
        visualization_keys.append(time_series_key)
        plt.close()
        
        # Correlation heatmap
        if len(df.columns) > 1:
            plt.figure(figsize=(10, 8))
            corr_matrix = df.corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title('Correlation Matrix')
            plt.tight_layout()
            
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
            img_buffer.seek(0)
            
            correlation_key = f"visualizations/{report_id}/correlation.png"
            s3_client.put_object(
                Bucket=s3_bucket,
                Key=correlation_key,
                Body=img_buffer.getvalue(),
                ContentType='image/png'
            )
            visualization_keys.append(correlation_key)
            plt.close()
        
        # Distribution plots
        for column in df.columns:
            if column != 'Date':
                plt.figure(figsize=(8, 6))
                plt.hist(df[column].dropna(), bins=30, alpha=0.7, edgecolor='black')
                plt.title(f'Distribution of {column}')
                plt.xlabel('Value')
                plt.ylabel('Frequency')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
                img_buffer.seek(0)
                
                dist_key = f"visualizations/{report_id}/distribution_{column}.png"
                s3_client.put_object(
                    Bucket=s3_bucket,
                    Key=dist_key,
                    Body=img_buffer.getvalue(),
                    ContentType='image/png'
                )
                visualization_keys.append(dist_key)
                plt.close()
    
    except Exception as e:
        logger.error(f"Error creating visualizations: {e}")
    
    return visualization_keys

def save_report_to_s3(report_data: Dict, s3_bucket: str, report_id: str) -> str:
    """Save report data to S3"""
    try:
        report_key = f"reports/{report_id}/report.json"
        
        s3_client.put_object(
            Bucket=s3_bucket,
            Key=report_key,
            Body=json.dumps(report_data, default=str),
            ContentType='application/json'
        )
        
        return report_key
    except Exception as e:
        logger.error(f"Error saving report to S3: {e}")
        raise

def lambda_handler(event: Dict, context) -> Dict:
    """Main Lambda handler function"""
    try:
        logger.info(f"Received event: {json.dumps(event)}")
        
        # Parse input
        if isinstance(event.get('body'), str):
            payload = json.loads(event['body'])
        else:
            payload = event
        
        indicators = payload.get('indicators', ['GDP', 'UNRATE', 'CPIAUCSL'])
        start_date = payload.get('start_date', (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'))
        end_date = payload.get('end_date', datetime.now().strftime('%Y-%m-%d'))
        options = payload.get('options', {})
        
        # Generate report ID
        report_id = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Processing analysis for indicators: {indicators}")
        logger.info(f"Date range: {start_date} to {end_date}")
        
        # Fetch data from FRED
        series_data = {}
        for indicator in indicators:
            if indicator in ECONOMIC_INDICATORS:
                series_id = ECONOMIC_INDICATORS[indicator]
                data = get_fred_data(series_id, start_date, end_date)
                if data is not None:
                    series_data[indicator] = data
                    logger.info(f"Successfully fetched data for {indicator}")
                else:
                    logger.warning(f"Failed to fetch data for {indicator}")
        
        # Create DataFrame
        df = create_dataframe(series_data)
        
        if df.empty:
            raise ValueError("No data available for analysis")
        
        # Generate analysis results
        report_data = {
            'report_id': report_id,
            'timestamp': datetime.now().isoformat(),
            'indicators': indicators,
            'start_date': start_date,
            'end_date': end_date,
            'total_observations': len(df),
            'data_shape': df.shape,
            'statistics': generate_statistics(df),
            'correlation_matrix': create_correlation_matrix(df),
            'data': df.reset_index().to_dict('records')
        }
        
        # Create visualizations if requested
        if options.get('visualizations', True):
            visualization_keys = create_visualizations(df, S3_BUCKET, report_id)
            report_data['visualizations'] = visualization_keys
        
        # Save report to S3
        report_key = save_report_to_s3(report_data, S3_BUCKET, report_id)
        
        logger.info(f"Analysis completed successfully. Report saved to: {report_key}")
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'status': 'success',
                'report_id': report_id,
                'report_key': report_key,
                'message': 'Analysis completed successfully'
            })
        }
        
    except Exception as e:
        logger.error(f"Error in lambda_handler: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'status': 'error',
                'message': str(e)
            })
        } 