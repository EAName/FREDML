---
tags:
- FRED
- ECONOMIC
- ML
---
# FRED ML - Federal Reserve Economic Data Machine Learning System

A comprehensive Machine Learning system for analyzing Federal Reserve Economic Data (FRED) with automated data processing, advanced analytics, and interactive visualizations.

## 🚀 Features

### Core Capabilities
- **📊 Real-time Data Processing**: Automated FRED API integration with enhanced client
- **🔍 Data Quality Assessment**: Comprehensive data validation and quality metrics
- **🔄 Automated Workflows**: CI/CD pipeline with quality gates
- **☁️ Cloud-Native**: AWS Lambda and S3 integration
- **🧪 Comprehensive Testing**: Unit, integration, and E2E tests

### Advanced Analytics
- **🤖 Statistical Modeling**: 
  - Linear regression with lagged variables
  - Correlation analysis (Pearson, Spearman, Kendall)
  - Granger causality testing
  - Comprehensive diagnostic testing (normality, homoscedasticity, autocorrelation, multicollinearity)
  - Principal Component Analysis (PCA)

- **🔮 Time Series Forecasting**:
  - ARIMA models with automatic order selection
  - Exponential Smoothing (ETS) models
  - Stationarity testing (ADF, KPSS)
  - Time series decomposition (trend, seasonal, residual)
  - Backtesting with performance metrics (MAE, RMSE, MAPE)
  - Confidence intervals and uncertainty quantification

- **🎯 Economic Segmentation**:
  - Time period clustering (economic regimes)
  - Series clustering (behavioral patterns)
  - K-means and hierarchical clustering
  - Optimal cluster detection (elbow method, silhouette analysis)
  - Dimensionality reduction (PCA, t-SNE)

- **📈 Interactive Visualizations**: Dynamic charts and dashboards
- **💡 Comprehensive Insights**: Automated insights extraction and key findings identification

## 📁 Project Structure

```
FRED_ML/
├── 📁 src/                    # Core application code
│   ├── 📁 core/              # Core pipeline components
│   ├── 📁 analysis/          # Economic analysis modules
│   ├── 📁 visualization/     # Data visualization components
│   └── 📁 lambda/           # AWS Lambda functions
├── 📁 scripts/               # Utility and demo scripts
│   ├── 📄 streamlit_demo.py  # Interactive Streamlit demo
│   ├── 📄 run_tests.py       # Test runner
│   └── 📄 simple_demo.py     # Command-line demo
├── 📁 tests/                 # Comprehensive test suite
│   ├── 📁 unit/             # Unit tests
│   ├── 📁 integration/      # Integration tests
│   └── 📁 e2e/              # End-to-end tests
├── 📁 docs/                  # Documentation
│   ├── 📁 api/              # API documentation
│   ├── 📁 architecture/     # System architecture docs
│   └── 📄 CONVERSATION_SUMMARY.md
├── 📁 config/               # Configuration files
├── 📁 data/                 # Data storage
│   ├── 📁 raw/             # Raw data files
│   ├── 📁 processed/       # Processed data
│   └── 📁 exports/         # Generated exports
├── 📁 deploy/               # Deployment configurations
│   ├── 📁 docker/          # Docker configurations
│   ├── 📁 kubernetes/      # Kubernetes manifests
│   └── 📁 helm/            # Helm charts
├── 📁 infrastructure/       # Infrastructure as code
│   ├── 📁 ci-cd/          # CI/CD configurations
│   ├── 📁 monitoring/      # Monitoring setup
│   └── 📁 alerts/          # Alert configurations
├── 📁 .github/workflows/    # GitHub Actions workflows
├── 📄 requirements.txt      # Python dependencies
├── 📄 pyproject.toml       # Project configuration
├── 📄 Dockerfile           # Container configuration
├── 📄 Makefile             # Build automation
└── 📄 README.md            # This file
```

## 🛠️ Quick Start

### Prerequisites

- Python 3.8+
- AWS Account (for cloud features)
- FRED API Key

### Installation

1. **Clone the repository**
   You can clone from any of the following remotes:
   ```bash
   # EAName GitHub
   git clone https://github.com/EAName/FREDML.git
   
   # ParallelLLC GitHub
   git clone https://github.com/ParallelLLC/FREDML.git

   # esalguero Hugging Face
   git clone https://huggingface.co/esalguero/FREDML

   # ParallelLLC Hugging Face
   git clone https://huggingface.co/ParallelLLC/FREDML
   ```
   cd FRED_ML
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   export AWS_ACCESS_KEY_ID="your_access_key"
   export AWS_SECRET_ACCESS_KEY="your_secret_key"
   export AWS_DEFAULT_REGION="us-east-1"
   export FRED_API_KEY="your_fred_api_key"
   ```

4. **Set up FRED API (Optional but Recommended)**
   ```bash
   # Run setup wizard
   python frontend/setup_fred.py
   
   # Test your FRED API key
   python frontend/test_fred_api.py
   ```

5. **Run the interactive demo**
   ```bash
   streamlit run scripts/streamlit_demo.py
   ```

## 🧪 Testing

### Run all tests
```bash
python scripts/run_tests.py
```

### Run specific test types
```bash
# Unit tests
python -m pytest tests/unit/

# Integration tests
python -m pytest tests/integration/

# End-to-end tests
python -m pytest tests/e2e/
```

### Development testing
```bash
python scripts/test_dev.py
```

## 🚀 Deployment

### Local Development
```bash
# Start development environment
python scripts/dev_setup.py

# Run development tests
python scripts/run_dev_tests.py
```

### Streamlit Cloud Deployment (Free)
```bash
# 1. Push to GitHub
git add .
git commit -m "Prepare for Streamlit Cloud deployment"
git push origin main

# 2. Deploy to Streamlit Cloud
# Go to https://share.streamlit.io/
# Connect your GitHub repository
# Set main file path to: streamlit_app.py
# Add environment variables for FRED_API_KEY and AWS credentials
```

### Production Deployment
```bash
# Deploy to AWS
python scripts/deploy_aws.py

# Deploy complete system
python scripts/deploy_complete.py
```

## 📊 Demo Applications

### Interactive Streamlit Demo
```bash
streamlit run scripts/streamlit_demo.py
```
Access at: http://localhost:8501

### Command-line Demo
```bash
python scripts/simple_demo.py
```

### Advanced Analytics Demo
```bash
# Run comprehensive analytics demo
python scripts/comprehensive_demo.py

# Run advanced analytics pipeline
python scripts/run_advanced_analytics.py --indicators GDPC1 INDPRO RSAFS --forecast-periods 4

# Run with custom parameters
python scripts/run_advanced_analytics.py \
  --indicators GDPC1 INDPRO RSAFS CPIAUCSL FEDFUNDS DGS10 \
  --start-date 2010-01-01 \
  --end-date 2024-01-01 \
  --forecast-periods 8 \
  --output-dir data/exports/advanced_analysis
```

## 🔧 Configuration

### Real vs Demo Data

The application supports two modes:

#### 🎯 Real FRED Data (Recommended)
- **Requires**: Free FRED API key from https://fred.stlouisfed.org/docs/api/api_key.html
- **Features**: Live economic data, real-time insights, actual forecasts
- **Setup**: 
  ```bash
  export FRED_API_KEY="your-actual-api-key"
  python frontend/test_fred_api.py  # Test your key
  ```

#### 📊 Demo Data (Fallback)
- **Features**: Realistic economic data for demonstration
- **Use case**: When API key is not available or for testing
- **Data**: Generated based on historical patterns and economic principles

### Environment Variables
- `AWS_ACCESS_KEY_ID`: AWS access key
- `AWS_SECRET_ACCESS_KEY`: AWS secret key
- `AWS_DEFAULT_REGION`: AWS region (default: us-east-1)
- `FRED_API_KEY`: FRED API key (get free key from FRED website)

### Configuration Files
- `config/pipeline.yaml`: Pipeline configuration
- `config/settings.py`: Application settings

## 📈 System Architecture

### Components
- **Frontend**: Streamlit interactive dashboard
- **Backend**: AWS Lambda serverless functions
- **Storage**: AWS S3 for data persistence
- **Scheduling**: EventBridge for automated triggers
- **Data Source**: FRED API for economic indicators

### Data Flow
```
FRED API → AWS Lambda → S3 Storage → Streamlit Dashboard
            ↓
        EventBridge (Scheduling)
            ↓
        CloudWatch (Monitoring)
```

## 🧪 Testing Strategy

### Test Types
- **Unit Tests**: Individual component testing
- **Integration Tests**: API and data flow testing
- **End-to-End Tests**: Complete system workflow testing

### Coverage
- Core pipeline components: 100%
- API integrations: 100%
- Data processing: 100%
- Visualization components: 100%

## 🔄 CI/CD Pipeline

### GitHub Actions Workflows
- **Main Pipeline**: Production deployments
- **Pull Request Checks**: Code quality validation
- **Scheduled Maintenance**: Automated updates
- **Release Management**: Version control

### Quality Gates
- Automated testing
- Code linting and formatting
- Security vulnerability scanning
- Documentation generation

## 📚 Documentation

- [API Documentation](docs/api/)
- [Architecture Guide](docs/architecture/)
- [Deployment Guide](docs/deployment/)
- [User Guide](docs/user-guide/)
- [Conversation Summary](docs/CONVERSATION_SUMMARY.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `python scripts/run_tests.py`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For support and questions:
- Create an issue on GitHub
- Check the [documentation](docs/)
- Review the [conversation summary](docs/CONVERSATION_SUMMARY.md)

---

**FRED ML** - Transforming economic data analysis with machine learning and automation.
