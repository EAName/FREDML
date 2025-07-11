---
tags:
- economic
- FRED
- FASTAPI
- ML
---
# FRED ML - Enterprise Economic Data Analysis Platform

A production-grade Python platform for collecting, analyzing, and visualizing Federal Reserve Economic Data (FRED) using the FRED API. Built with enterprise-grade architecture including FastAPI, Docker, Kubernetes, and comprehensive monitoring.

## Features

- **Production-Ready API**: FastAPI-based REST API with automatic documentation
- **Containerized Deployment**: Docker and Docker Compose for easy deployment
- **Kubernetes Support**: Helm charts and K8s manifests for cloud deployment
- **Monitoring & Observability**: Prometheus metrics and structured logging
- **Data Collection**: Fetch economic indicators from FRED API
- **Advanced Analytics**: Machine learning models and statistical analysis
- **Visualization**: Create time series plots and charts
- **Data Export**: Save data to CSV format
- **Flexible Configuration**: Environment-based configuration
- **Comprehensive Testing**: Unit, integration, and E2E tests
- **CI/CD Ready**: Pre-commit hooks and automated quality checks

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
├── src/                     # Source code
│   ├── core/               # Core functionality
│   ├── analysis/           # Analysis modules
│   ├── utils/              # Utility functions
│   └── visualization/      # Visualization modules
├── config/                  # Configuration settings
│   ├── settings.py         # Environment variables and settings
│   └── pipeline.yaml       # Pipeline configuration
├── deployment/              # Deployment configurations
├── docker/                  # Docker configurations
├── kubernetes/              # K8s manifests
├── helm/                    # Helm charts
├── scripts/                 # Executable scripts
│   ├── dev/                # Development scripts
│   ├── prod/               # Production scripts
│   └── deploy/             # Deployment scripts
├── tests/                   # Test files
│   ├── unit/               # Unit tests
│   ├── integration/        # Integration tests
│   └── e2e/                # End-to-end tests
├── docs/                    # Documentation
│   ├── api/                # API documentation
│   ├── user_guide/         # User guides
│   ├── deployment/         # Deployment guides
│   └── architecture/       # Architecture docs
├── monitoring/              # Monitoring configurations
├── alerts/                  # Alert configurations
├── data/                    # Data directories
│   ├── raw/                # Raw data
│   ├── processed/          # Processed data
│   └── exports/            # Exported files
├── logs/                    # Application logs
├── requirements.txt         # Python dependencies
├── Dockerfile              # Docker image
├── docker-compose.yml      # Local development
├── Makefile                # Build automation
├── .env.example            # Environment variables template
├── .pre-commit-config.yaml # Code quality hooks
└── README.md               # This file
```

## Usage

### Basic Usage

#### Local Development

Run the application locally:

```bash
make run
```

Or with Docker Compose:

```bash
make run-docker
```

#### API Usage

Once running, access the API at `http://localhost:8000`:

- **API Documentation**: `http://localhost:8000/docs`
- **Health Check**: `http://localhost:8000/health`
- **Available Indicators**: `http://localhost:8000/api/v1/indicators`

#### Scripts

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

### Environment Variables

The application uses environment variables for configuration:

- `FRED_API_KEY`: Your FRED API key (required)
- `ENVIRONMENT`: `development` or `production` (default: development)
- `PORT`: Application port (default: 8000)
- `POSTGRES_PASSWORD`: Database password for Docker Compose

### Customization

Edit `config/settings.py` to customize:
- Default date ranges
- Output directories
- Default indicators

## Dependencies

### Core Dependencies
- `fredapi`: FRED API client
- `pandas`: Data manipulation
- `numpy`: Numerical computing
- `matplotlib`: Plotting
- `seaborn`: Statistical visualization
- `scikit-learn`: Machine learning
- `statsmodels`: Statistical models

### Production Dependencies
- `fastapi`: Web framework
- `uvicorn`: ASGI server
- `redis`: Caching
- `psycopg2-binary`: PostgreSQL adapter
- `sqlalchemy`: ORM
- `prometheus-client`: Metrics

### Development Dependencies
- `pytest`: Testing framework
- `black`: Code formatting
- `flake8`: Linting
- `mypy`: Type checking
- `pre-commit`: Git hooks

## Error Handling

The tool includes comprehensive error handling for:
- API connection issues
- Invalid series IDs
- Rate limit exceeded
- Data format errors

## Development

### Setup Development Environment

```bash
make setup-dev
```

### Code Quality

```bash
make format    # Format code
make lint      # Run linting
make test      # Run tests
```

### Deployment

```bash
make build     # Build Docker image
make deploy    # Deploy to Kubernetes
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting: `make test lint`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **Documentation**: Check the `docs/` directory
- **Issues**: Report bugs via GitHub Issues
- **FRED API**: https://fred.stlouisfed.org/docs/api/