# FRED Economic Data Analysis Tool

A comprehensive Python tool for collecting, analyzing, and visualizing Federal Reserve Economic Data (FRED) using the FRED API.

## Features

- **Data Collection**: Fetch economic indicators from FRED API
- **Data Analysis**: Generate summary statistics and insights
- **Visualization**: Create time series plots and charts
- **Data Export**: Save data to CSV format
- **Flexible Configuration**: Easy customization of indicators and date ranges

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. API Key Configuration

1. Get your FRED API key from [FRED API](https://fred.stlouisfed.org/docs/api/api_key.html)
2. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
3. Edit `.env` and add your API key:
   ```
   FRED_API_KEY=your_actual_api_key_here
   ```

### 3. Project Structure

```
FRED_ML/
├── config/               # Configuration settings
│   ├── settings.py       # Environment variables and settings
│   └── pipeline.yaml     # Pipeline configuration
├── src/                  # Source code
│   ├── core/            # Core functionality
│   ├── analysis/        # Analysis modules
│   ├── utils/           # Utility functions
│   └── visualization/   # Visualization modules
├── scripts/             # Executable scripts
├── tests/               # Test files
├── data/                # Data directories
│   ├── raw/            # Raw data
│   ├── processed/      # Processed data
│   └── exports/        # Exported files
├── requirements.txt     # Python dependencies
├── .env.example        # Environment variables template
└── README.md           # This file
```

## Usage

### Basic Usage

Run the EDA script to perform exploratory data analysis:

```bash
python scripts/run_eda.py
```

Or run the advanced analytics:

```bash
python scripts/run_advanced_analytics.py
```

This will:
- Fetch data for key economic indicators (GDP, Unemployment Rate, CPI, Federal Funds Rate, 10-Year Treasury Rate)
- Generate summary statistics
- Create visualizations
- Save data to CSV files

### Custom Analysis

You can customize the analysis by importing the modules:

```python
from src.core.fred_client import FREDDataCollectorV2
from src.analysis.advanced_analytics import AdvancedAnalytics

# Initialize collector
collector = FREDDataCollectorV2()

# Custom series and date range
custom_series = ['GDP', 'UNRATE', 'CPIAUCSL']
start_date = '2020-01-01'
end_date = '2024-01-01'

# Run analysis
df, summary = collector.run_analysis(
    series_ids=custom_series,
    start_date=start_date,
    end_date=end_date
)
```

## Available Economic Indicators

The tool includes these common economic indicators:

| Series ID | Description |
|-----------|-------------|
| GDP | Gross Domestic Product |
| UNRATE | Unemployment Rate |
| CPIAUCSL | Consumer Price Index |
| FEDFUNDS | Federal Funds Rate |
| DGS10 | 10-Year Treasury Rate |
| DEXUSEU | US/Euro Exchange Rate |
| PAYEMS | Total Nonfarm Payrolls |
| INDPRO | Industrial Production |
| M2SL | M2 Money Stock |
| PCE | Personal Consumption Expenditures |

## Output Files

### Data Files
- CSV files saved in the `data/` directory
- Timestamped filenames (e.g., `fred_economic_data_20241201_143022.csv`)

### Visualization Files
- PNG plots saved in the `plots/` directory
- High-resolution charts with economic indicator time series

## API Rate Limits

The FRED API has rate limits:
- 120 requests per minute
- 1000 requests per day

The tool includes error handling for rate limit issues.

## Configuration

Edit `config/settings.py` to customize:
- Default date ranges
- Output directories
- Default indicators

The API key is now managed through environment variables (see Setup section above).

## Dependencies

- `fredapi`: FRED API client
- `pandas`: Data manipulation
- `numpy`: Numerical computing
- `matplotlib`: Plotting
- `seaborn`: Statistical visualization
- `jupyter`: Interactive notebooks (optional)

## Error Handling

The tool includes comprehensive error handling for:
- API connection issues
- Invalid series IDs
- Rate limit exceeded
- Data format errors

## Contributing

To add new features:
1. Extend the `FREDDataCollector` class
2. Add new methods for specific analysis
3. Update the configuration as needed

## License

This project is for educational and research purposes. Please respect FRED API terms of service.

## Support

For issues with the FRED API, visit: https://fred.stlouisfed.org/docs/api/ 