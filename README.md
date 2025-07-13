# FRED ML - Real-Time Economic Analytics Platform

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)

A comprehensive real-time economic analytics platform that leverages the Federal Reserve Economic Data (FRED) API to provide advanced economic insights, forecasting, and visualization capabilities.

## 🚀 Features

### 📊 Real-Time Economic Data
- **Live FRED API Integration**: Direct connection to Federal Reserve Economic Data
- **12+ Economic Indicators**: GDP, CPI, Unemployment, Industrial Production, and more
- **Real-Time Updates**: Latest economic data with automatic refresh
- **Data Validation**: Robust error handling and data quality checks

### 🔮 Advanced Analytics
- **Economic Forecasting**: Time series analysis and predictive modeling
- **Correlation Analysis**: Spearman correlations with z-score standardization
- **Growth Rate Analysis**: Year-over-year and period-over-period calculations
- **Statistical Modeling**: Comprehensive statistical analysis and insights

### 📈 Interactive Visualizations
- **Time Series Charts**: Dynamic economic indicator trends
- **Correlation Heatmaps**: Interactive correlation matrices
- **Distribution Analysis**: Statistical distribution visualizations
- **Forecast Plots**: Predictive modeling visualizations

### 🎯 Key Insights
- **Economic Health Scoring**: Real-time economic health assessment
- **Market Sentiment Analysis**: Bullish/bearish market indicators
- **Risk Factor Analysis**: Comprehensive risk assessment
- **Opportunity Identification**: Strategic opportunity analysis

### 📥 Data Export & Downloads
- **CSV Export**: Raw economic data downloads
- **Excel Reports**: Multi-sheet analysis reports
- **Bulk Downloads**: Complete data packages
- **Visualization Downloads**: High-quality chart exports

## 🛠️ Technology Stack

- **Frontend**: Streamlit (Python web framework)
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib
- **API Integration**: FRED API (Federal Reserve Economic Data)
- **Cloud Storage**: AWS S3 (optional)
- **Deployment**: Docker, Hugging Face Spaces

## 📋 Prerequisites

- Python 3.11 or higher
- FRED API key (free from [FRED](https://fred.stlouisfed.org/docs/api/api_key.html))
- Git

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/FRED_ML.git
cd FRED_ML
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables
Create a `.env` file in the project root:
```bash
FRED_API_KEY=your_fred_api_key_here
```

Or set the environment variable directly:
```bash
export FRED_API_KEY=your_fred_api_key_here
```

### 4. Get Your FRED API Key
1. Visit [FRED API Key Registration](https://fred.stlouisfed.org/docs/api/api_key.html)
2. Sign up for a free account
3. Generate your API key
4. Add it to your environment variables

## 🎯 Quick Start

### Local Development
```bash
# Start the Streamlit app
streamlit run frontend/app.py --server.port 8501
```

### Docker Deployment
```bash
# Build the Docker image
docker build -t fred-ml .

# Run the container
docker run -p 8501:8501 -e FRED_API_KEY=your_key_here fred-ml
```

### Hugging Face Spaces
The app is automatically deployed to Hugging Face Spaces and can be accessed at:
[FRED ML on Hugging Face](https://huggingface.co/spaces/yourusername/fred-ml)

## 📖 Usage

### 1. Executive Dashboard
- **Real-time economic metrics**
- **Key performance indicators**
- **Economic health scoring**
- **Market sentiment analysis**

### 2. Economic Indicators
- **Interactive data exploration**
- **Real-time data validation**
- **Growth rate analysis**
- **Statistical insights**

### 3. Advanced Analytics
- **Comprehensive analysis options**
- **Forecasting capabilities**
- **Segmentation analysis**
- **Statistical modeling**

### 4. Reports & Insights
- **Real-time economic insights**
- **Generated reports**
- **Market analysis**
- **Risk assessment**

### 5. Downloads
- **Data export capabilities**
- **Visualization downloads**
- **Bulk data packages**
- **Report generation**

## 🔧 Configuration

### Environment Variables
| Variable | Description | Required |
|----------|-------------|----------|
| `FRED_API_KEY` | Your FRED API key | Yes |
| `AWS_ACCESS_KEY_ID` | AWS access key (for S3) | No |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key (for S3) | No |

### API Configuration
The app supports various FRED API endpoints:
- **Economic Indicators**: GDP, CPI, Unemployment, etc.
- **Financial Data**: Treasury yields, Federal Funds Rate
- **Employment Data**: Nonfarm payrolls, labor statistics
- **Production Data**: Industrial production, capacity utilization

## 📊 Data Sources

### Primary Economic Indicators
- **GDPC1**: Real Gross Domestic Product
- **CPIAUCSL**: Consumer Price Index
- **UNRATE**: Unemployment Rate
- **INDPRO**: Industrial Production
- **FEDFUNDS**: Federal Funds Rate
- **DGS10**: 10-Year Treasury Constant Maturity Rate
- **RSAFS**: Retail Sales
- **PAYEMS**: Total Nonfarm Payrolls
- **PCE**: Personal Consumption Expenditures
- **M2SL**: M2 Money Stock
- **TCU**: Capacity Utilization
- **DEXUSEU**: US/Euro Exchange Rate

## 🏗️ Project Structure

```
FRED_ML/
├── frontend/                 # Streamlit application
│   ├── app.py               # Main application file
│   ├── fred_api_client.py   # FRED API integration
│   └── demo_data.py         # Demo data generation
├── src/                     # Core analytics engine
│   ├── core/               # Core data processing
│   ├── analysis/           # Analytics modules
│   └── visualization/      # Chart generation
├── tests/                  # Test suite
├── requirements.txt         # Python dependencies
├── Dockerfile              # Docker configuration
├── README.md              # This file
└── LICENSE                # Apache 2.0 License
```

## 🧪 Testing

### Run All Tests
```bash
python -m pytest tests/
```

### Run Specific Test Categories
```bash
# Test FRED API integration
python -m pytest tests/test_fred_api.py

# Test analytics functionality
python -m pytest tests/test_analytics.py

# Test data processing
python -m pytest tests/test_data_processing.py
```

## 🚀 Deployment

### Local Development
```bash
streamlit run frontend/app.py --server.port 8501
```

### Docker Deployment
```bash
docker build -t fred-ml .
docker run -p 8501:8501 -e FRED_API_KEY=your_key_here fred-ml
```

### Hugging Face Spaces
1. Fork this repository
2. Create a new Space on Hugging Face
3. Connect your repository
4. Set environment variables in Space settings

### AWS Deployment
```bash
# Deploy to AWS Lambda
sam build
sam deploy --guided
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

### Code Style
- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings to functions
- Include unit tests for new features

## 📝 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Federal Reserve Bank of St. Louis** for providing the FRED API
- **Streamlit** for the excellent web framework
- **Pandas & NumPy** for data processing capabilities
- **Plotly** for interactive visualizations

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/FRED_ML/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/FRED_ML/discussions)
- **Documentation**: [Wiki](https://github.com/yourusername/FRED_ML/wiki)

## 🔗 Links

- **Live Demo**: [FRED ML on Hugging Face](https://huggingface.co/spaces/yourusername/fred-ml)
- **FRED API**: [Federal Reserve Economic Data](https://fred.stlouisfed.org/)
- **Documentation**: [Project Wiki](https://github.com/yourusername/FRED_ML/wiki)

---

**Made with ❤️ for economic data enthusiasts**
