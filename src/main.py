#!/usr/bin/env python3
"""
FRED ML - Main Application Entry Point
Production-grade FastAPI application for economic data analysis
"""

import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config.settings import FRED_API_KEY
from src.analysis.advanced_analytics import AdvancedAnalytics
from src.core.fred_client import FREDDataCollectorV2

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global variables for application state
collector = None
analytics = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    global collector, analytics
    logger.info("Starting FRED ML application...")

    if not FRED_API_KEY:
        logger.error("FRED_API_KEY not configured")
        raise ValueError("FRED_API_KEY environment variable is required")

    collector = FREDDataCollectorV2(api_key=FRED_API_KEY)
    logger.info("FRED Data Collector initialized")

    yield

    # Shutdown
    logger.info("Shutting down FRED ML application...")


# Create FastAPI application
app = FastAPI(
    title="FRED ML API",
    description="Economic Data Analysis API using Federal Reserve Economic Data",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "FRED ML API", "version": "1.0.0", "status": "running"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint"""
    if collector is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    return {"status": "ready"}


@app.get("/api/v1/indicators")
async def get_indicators():
    """Get available economic indicators"""
    if collector is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    return {
        "indicators": list(collector.indicators.keys()),
        "descriptions": collector.indicators,
    }


@app.post("/api/v1/analyze")
async def analyze_data(
    series_ids: list[str], start_date: str = None, end_date: str = None
):
    """Analyze economic data for specified series"""
    if collector is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    try:
        df, summary = collector.run_analysis(
            series_ids=series_ids, start_date=start_date, end_date=end_date
        )

        return {
            "status": "success",
            "data_shape": df.shape if df is not None else None,
            "summary": summary.to_dict() if summary is not None else None,
        }
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/status")
async def get_status():
    """Get application status"""
    return {
        "api_key_configured": bool(FRED_API_KEY),
        "collector_initialized": collector is not None,
        "environment": os.getenv("ENVIRONMENT", "development"),
    }


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=port,
        reload=os.getenv("ENVIRONMENT") == "development",
    )
