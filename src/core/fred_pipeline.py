from .base_pipeline import BasePipeline
import requests
import pandas as pd
import os
from datetime import datetime

class FREDPipeline(BasePipeline):
    """
    FRED Data Pipeline: Extracts, transforms, and loads FRED data using config.
    """
    def __init__(self, config_path: str):
        super().__init__(config_path)
        self.fred_cfg = self.config['fred']
        self.api_key = self.fred_cfg['api_key']
        self.series = self.fred_cfg['series']
        self.start_date = self.fred_cfg['start_date']
        self.end_date = self.fred_cfg['end_date']
        self.output_dir = self.fred_cfg['output_dir']
        self.export_dir = self.fred_cfg['export_dir']
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.export_dir, exist_ok=True)

    def extract(self):
        """Extract data from FRED API for all configured series."""
        base_url = "https://api.stlouisfed.org/fred/series/observations"
        data = {}
        for series_id in self.series:
            params = {
                'series_id': series_id,
                'api_key': self.api_key,
                'file_type': 'json',
                'start_date': self.start_date,
                'end_date': self.end_date
            }
            try:
                resp = requests.get(base_url, params=params)
                resp.raise_for_status()
                obs = resp.json().get('observations', [])
                dates, values = [], []
                for o in obs:
                    try:
                        dates.append(pd.to_datetime(o['date']))
                        values.append(float(o['value']) if o['value'] != '.' else None)
                    except Exception:
                        continue
                data[series_id] = pd.Series(values, index=dates, name=series_id)
                self.logger.info(f"Extracted {len(values)} records for {series_id}")
            except Exception as e:
                self.logger.error(f"Failed to extract {series_id}: {e}")
        return data

    def transform(self, data):
        """Transform raw data into a DataFrame, align dates, handle missing."""
        if not data:
            self.logger.warning("No data to transform.")
            return pd.DataFrame()
        all_dates = set()
        for s in data.values():
            all_dates.update(s.index)
        if not all_dates:
            return pd.DataFrame()
        date_range = pd.date_range(min(all_dates), max(all_dates), freq='D')
        df = pd.DataFrame(index=date_range)
        for k, v in data.items():
            df[k] = v
        df.index.name = 'Date'
        self.logger.info(f"Transformed data to DataFrame with shape {df.shape}")
        return df

    def load(self, df):
        """Save DataFrame to CSV in output_dir and export_dir."""
        if df.empty:
            self.logger.warning("No data to load.")
            return None
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(self.output_dir, f'fred_data_{ts}.csv')
        exp_path = os.path.join(self.export_dir, f'fred_data_{ts}.csv')
        df.to_csv(out_path)
        df.to_csv(exp_path)
        self.logger.info(f"Saved data to {out_path} and {exp_path}")
        return out_path, exp_path

    def run(self):
        self.logger.info("Starting FRED data pipeline run...")
        data = self.extract()
        df = self.transform(data)
        self.load(df)
        self.logger.info("FRED data pipeline run complete.") 