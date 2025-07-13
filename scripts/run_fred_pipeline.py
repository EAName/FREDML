#!/usr/bin/env python
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from core.fred_pipeline import FREDPipeline

if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'pipeline.yaml')
    pipeline = FREDPipeline(config_path)
    pipeline.run() 