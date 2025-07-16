#!/usr/bin/env python3
"""
Enterprise-grade configuration management for FRED ML
Centralized configuration with environment variable support and validation
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import logging
from datetime import datetime

# Constants for backward compatibility
DEFAULT_START_DATE = "2020-01-01"
DEFAULT_END_DATE = "2024-12-31"
FRED_API_KEY = os.getenv('FRED_API_KEY', '')
OUTPUT_DIR = "data/processed"
PLOTS_DIR = "data/exports"


@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    host: str = "localhost"
    port: int = 5432
    database: str = "fred_ml"
    username: str = "postgres"
    password: str = ""
    pool_size: int = 10
    max_overflow: int = 20
    echo: bool = False


@dataclass
class APIConfig:
    """API configuration settings"""
    fred_api_key: str = ""
    fred_base_url: str = "https://api.stlouisfed.org/fred"
    request_timeout: int = 30
    max_retries: int = 3
    rate_limit_delay: float = 0.1


@dataclass
class AWSConfig:
    """AWS configuration settings"""
    access_key_id: str = ""
    secret_access_key: str = ""
    region: str = "us-east-1"
    s3_bucket: str = "fred-ml-data"
    lambda_function: str = "fred-ml-analysis"
    cloudwatch_log_group: str = "/aws/lambda/fred-ml-analysis"


@dataclass
class LoggingConfig:
    """Logging configuration settings"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: str = "logs/fred_ml.log"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    console_output: bool = True
    file_output: bool = True


@dataclass
class AnalyticsConfig:
    """Analytics configuration settings"""
    output_directory: str = "data/analytics"
    cache_directory: str = "data/cache"
    max_data_points: int = 10000
    default_forecast_periods: int = 12
    confidence_level: float = 0.95
    enable_caching: bool = True
    cache_ttl: int = 3600  # 1 hour


@dataclass
class SecurityConfig:
    """Security configuration settings"""
    enable_ssl: bool = True
    allowed_origins: List[str] = field(default_factory=lambda: ["*"])
    api_rate_limit: int = 1000  # requests per hour
    session_timeout: int = 3600  # 1 hour
    enable_audit_logging: bool = True


@dataclass
class PerformanceConfig:
    """Performance configuration settings"""
    max_workers: int = 4
    chunk_size: int = 1000
    memory_limit: int = 1024 * 1024 * 1024  # 1GB
    enable_profiling: bool = False
    cache_size: int = 1000


class Config:
    """Enterprise-grade configuration manager for FRED ML"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file
        self.database = DatabaseConfig()
        self.api = APIConfig()
        self.aws = AWSConfig()
        self.logging = LoggingConfig()
        self.analytics = AnalyticsConfig()
        self.security = SecurityConfig()
        self.performance = PerformanceConfig()
        
        # Load configuration
        self._load_environment_variables()
        if config_file:
            self._load_config_file()
        
        # Validate configuration
        self._validate_config()
        
        # Setup logging
        self._setup_logging()
    
    def _load_environment_variables(self):
        """Load configuration from environment variables"""
        # Database configuration
        self.database.host = os.getenv("DB_HOST", self.database.host)
        self.database.port = int(os.getenv("DB_PORT", str(self.database.port)))
        self.database.database = os.getenv("DB_NAME", self.database.database)
        self.database.username = os.getenv("DB_USER", self.database.username)
        self.database.password = os.getenv("DB_PASSWORD", self.database.password)
        
        # API configuration
        self.api.fred_api_key = os.getenv("FRED_API_KEY", self.api.fred_api_key)
        self.api.fred_base_url = os.getenv("FRED_BASE_URL", self.api.fred_base_url)
        self.api.request_timeout = int(os.getenv("API_TIMEOUT", str(self.api.request_timeout)))
        
        # AWS configuration
        self.aws.access_key_id = os.getenv("AWS_ACCESS_KEY_ID", self.aws.access_key_id)
        self.aws.secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY", self.aws.secret_access_key)
        self.aws.region = os.getenv("AWS_DEFAULT_REGION", self.aws.region)
        self.aws.s3_bucket = os.getenv("AWS_S3_BUCKET", self.aws.s3_bucket)
        
        # Logging configuration
        self.logging.level = os.getenv("LOG_LEVEL", self.logging.level)
        self.logging.file_path = os.getenv("LOG_FILE", self.logging.file_path)
        
        # Analytics configuration
        self.analytics.output_directory = os.getenv("ANALYTICS_OUTPUT_DIR", self.analytics.output_directory)
        self.analytics.cache_directory = os.getenv("CACHE_DIR", self.analytics.cache_directory)
        
        # Performance configuration
        self.performance.max_workers = int(os.getenv("MAX_WORKERS", str(self.performance.max_workers)))
        self.performance.memory_limit = int(os.getenv("MEMORY_LIMIT", str(self.performance.memory_limit)))
    
    def _load_config_file(self):
        """Load configuration from file (if provided)"""
        if not self.config_file or not os.path.exists(self.config_file):
            return
        
        try:
            import yaml
            with open(self.config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Update configuration sections
            if 'database' in config_data:
                for key, value in config_data['database'].items():
                    if hasattr(self.database, key):
                        setattr(self.database, key, value)
            
            if 'api' in config_data:
                for key, value in config_data['api'].items():
                    if hasattr(self.api, key):
                        setattr(self.api, key, value)
            
            if 'aws' in config_data:
                for key, value in config_data['aws'].items():
                    if hasattr(self.aws, key):
                        setattr(self.aws, key, value)
            
            if 'logging' in config_data:
                for key, value in config_data['logging'].items():
                    if hasattr(self.logging, key):
                        setattr(self.logging, key, value)
            
            if 'analytics' in config_data:
                for key, value in config_data['analytics'].items():
                    if hasattr(self.analytics, key):
                        setattr(self.analytics, key, value)
            
            if 'security' in config_data:
                for key, value in config_data['security'].items():
                    if hasattr(self.security, key):
                        setattr(self.security, key, value)
            
            if 'performance' in config_data:
                for key, value in config_data['performance'].items():
                    if hasattr(self.performance, key):
                        setattr(self.performance, key, value)
                        
        except Exception as e:
            logging.warning(f"Failed to load config file {self.config_file}: {e}")
    
    def _validate_config(self):
        """Validate configuration settings"""
        errors = []
        
        # Validate required settings - make FRED_API_KEY optional for development
        if not self.api.fred_api_key:
            if os.getenv("ENVIRONMENT", "development").lower() == "production":
                errors.append("FRED_API_KEY is required in production")
            else:
                # In development, just warn but don't fail
                logging.warning("FRED_API_KEY not configured - some features will be limited")
        
        # AWS credentials are optional for cloud features
        if not self.aws.access_key_id and not self.aws.secret_access_key:
            logging.info("AWS credentials not configured - cloud features will be disabled")
        
        # Validate numeric ranges
        if self.api.request_timeout < 1 or self.api.request_timeout > 300:
            errors.append("API timeout must be between 1 and 300 seconds")
        
        if self.performance.max_workers < 1 or self.performance.max_workers > 32:
            errors.append("Max workers must be between 1 and 32")
        
        if self.analytics.confidence_level < 0.5 or self.analytics.confidence_level > 0.99:
            errors.append("Confidence level must be between 0.5 and 0.99")
        
        # Validate file paths
        if self.logging.file_path:
            log_dir = os.path.dirname(self.logging.file_path)
            if log_dir and not os.path.exists(log_dir):
                try:
                    os.makedirs(log_dir, exist_ok=True)
                except Exception as e:
                    errors.append(f"Cannot create log directory {log_dir}: {e}")
        
        if self.analytics.output_directory and not os.path.exists(self.analytics.output_directory):
            try:
                os.makedirs(self.analytics.output_directory, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create analytics output directory {self.analytics.output_directory}: {e}")
        
        if errors:
            raise ValueError(f"Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors))
    
    def _setup_logging(self):
        """Setup logging configuration"""
        # Create log directory if it doesn't exist
        if self.logging.file_path:
            log_dir = os.path.dirname(self.logging.file_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, self.logging.level.upper()),
            format=self.logging.format,
            handlers=self._get_log_handlers()
        )
    
    def _get_log_handlers(self) -> List[logging.Handler]:
        """Get log handlers based on configuration"""
        handlers = []
        
        if self.logging.console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(logging.Formatter(self.logging.format))
            handlers.append(console_handler)
        
        if self.logging.file_output and self.logging.file_path:
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                self.logging.file_path,
                maxBytes=self.logging.max_file_size,
                backupCount=self.logging.backup_count
            )
            file_handler.setFormatter(logging.Formatter(self.logging.format))
            handlers.append(file_handler)
        
        return handlers
    
    def get_fred_api_key(self) -> str:
        """Get FRED API key with validation"""
        if not self.api.fred_api_key:
            raise ValueError("FRED_API_KEY is not configured")
        return self.api.fred_api_key
    
    def get_database_url(self) -> str:
        """Get database connection URL"""
        if self.database.password:
            return f"postgresql://{self.database.username}:{self.database.password}@{self.database.host}:{self.database.port}/{self.database.database}"
        else:
            return f"postgresql://{self.database.username}@{self.database.host}:{self.database.port}/{self.database.database}"
    
    def get_aws_credentials(self) -> Dict[str, str]:
        """Get AWS credentials"""
        if not self.aws.access_key_id or not self.aws.secret_access_key:
            raise ValueError("AWS credentials are not configured")
        
        return {
            "aws_access_key_id": self.aws.access_key_id,
            "aws_secret_access_key": self.aws.secret_access_key,
            "region_name": self.aws.region
        }
    
    def is_production(self) -> bool:
        """Check if running in production mode"""
        return os.getenv("ENVIRONMENT", "development").lower() == "production"
    
    def is_development(self) -> bool:
        """Check if running in development mode"""
        return os.getenv("ENVIRONMENT", "development").lower() == "development"
    
    def get_cache_directory(self) -> str:
        """Get cache directory path"""
        if not os.path.exists(self.analytics.cache_directory):
            os.makedirs(self.analytics.cache_directory, exist_ok=True)
        return self.analytics.cache_directory
    
    def get_output_directory(self) -> str:
        """Get output directory path"""
        if not os.path.exists(self.analytics.output_directory):
            os.makedirs(self.analytics.output_directory, exist_ok=True)
        return self.analytics.output_directory
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "database": self.database.__dict__,
            "api": self.api.__dict__,
            "aws": self.aws.__dict__,
            "logging": self.logging.__dict__,
            "analytics": self.analytics.__dict__,
            "security": self.security.__dict__,
            "performance": self.performance.__dict__
        }
    
    def __str__(self) -> str:
        """String representation of configuration"""
        return f"Config(environment={os.getenv('ENVIRONMENT', 'development')}, fred_api_key={'*' * 8 if self.api.fred_api_key else 'Not set'})"


# Global configuration instance
_config_instance: Optional[Config] = None


def get_config() -> Config:
    """Get global configuration instance"""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance


def reload_config(config_file: Optional[str] = None) -> Config:
    """Reload configuration from file"""
    global _config_instance
    _config_instance = Config(config_file)
    return _config_instance


# Convenience functions for common configuration access
def get_fred_api_key() -> str:
    """Get FRED API key"""
    return get_config().get_fred_api_key()


def get_database_url() -> str:
    """Get database URL"""
    return get_config().get_database_url()


def get_aws_credentials() -> Dict[str, str]:
    """Get AWS credentials"""
    return get_config().get_aws_credentials()


def is_production() -> bool:
    """Check if running in production"""
    return get_config().is_production()


def is_development() -> bool:
    """Check if running in development"""
    return get_config().is_development() 