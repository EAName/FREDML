#!/usr/bin/env python3
"""
Enterprise-grade health check system for FRED ML
Comprehensive monitoring of all system components
"""

import sys
import os
import time
import json
import requests
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class HealthChecker:
    """Enterprise-grade health checker for FRED ML"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.health_results = {}
        self.start_time = time.time()
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging for health checks"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def check_python_environment(self) -> Dict[str, Any]:
        """Check Python environment health"""
        self.logger.info("Checking Python environment...")
        
        try:
            import sys
            import platform
            
            result = {
                "python_version": sys.version,
                "platform": platform.platform(),
                "architecture": platform.architecture(),
                "processor": platform.processor(),
                "status": "healthy"
            }
            
            # Check Python version
            if sys.version_info >= (3, 9):
                result["python_version_ok"] = True
            else:
                result["python_version_ok"] = False
                result["status"] = "warning"
                result["message"] = "Python version should be 3.9+"
            
            # Check virtual environment
            if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
                result["virtual_env"] = True
                result["virtual_env_path"] = sys.prefix
            else:
                result["virtual_env"] = False
                result["status"] = "warning"
                result["message"] = "Not running in virtual environment"
            
            self.logger.info("Python environment check completed")
            return result
            
        except Exception as e:
            self.logger.error(f"Python environment check failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def check_dependencies(self) -> Dict[str, Any]:
        """Check installed dependencies"""
        self.logger.info("Checking dependencies...")
        
        try:
            import pkg_resources
            import subprocess
            
            # Get installed packages
            installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
            
            # Check required packages
            required_packages = [
                "pandas", "numpy", "matplotlib", "seaborn", "streamlit",
                "requests", "scikit-learn", "scipy", "statsmodels"
            ]
            
            missing_packages = []
            outdated_packages = []
            
            for package in required_packages:
                if package not in installed_packages:
                    missing_packages.append(package)
                else:
                    # Could add version checking here
                    pass
            
            result = {
                "installed_packages": len(installed_packages),
                "required_packages": len(required_packages),
                "missing_packages": missing_packages,
                "outdated_packages": outdated_packages,
                "status": "healthy" if not missing_packages else "warning"
            }
            
            if missing_packages:
                result["message"] = f"Missing packages: {', '.join(missing_packages)}"
            
            self.logger.info("Dependencies check completed")
            return result
            
        except Exception as e:
            self.logger.error(f"Dependencies check failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def check_configuration(self) -> Dict[str, Any]:
        """Check configuration health"""
        self.logger.info("Checking configuration...")
        
        try:
            from config.settings import get_config
            
            config = get_config()
            
            result = {
                "fred_api_key_configured": bool(config.api.fred_api_key),
                "aws_configured": bool(config.aws.access_key_id and config.aws.secret_access_key),
                "environment": os.getenv("ENVIRONMENT", "development"),
                "log_level": config.logging.level,
                "status": "healthy"
            }
            
            # Check for required configuration
            if not result["fred_api_key_configured"]:
                result["status"] = "warning"
                result["message"] = "FRED API key not configured"
            
            if not result["aws_configured"]:
                result["status"] = "warning"
                result["message"] = "AWS credentials not configured (cloud features disabled)"
            
            self.logger.info("Configuration check completed")
            return result
            
        except Exception as e:
            self.logger.error(f"Configuration check failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def check_file_system(self) -> Dict[str, Any]:
        """Check file system health"""
        self.logger.info("Checking file system...")
        
        try:
            import shutil
            
            result = {
                "project_root_exists": self.project_root.exists(),
                "src_directory_exists": (self.project_root / "src").exists(),
                "tests_directory_exists": (self.project_root / "tests").exists(),
                "config_directory_exists": (self.project_root / "config").exists(),
                "data_directory_exists": (self.project_root / "data").exists(),
                "logs_directory_exists": (self.project_root / "logs").exists(),
                "status": "healthy"
            }
            
            # Check disk space
            try:
                disk_usage = shutil.disk_usage(self.project_root)
                result["disk_free_gb"] = disk_usage.free / (1024**3)
                result["disk_total_gb"] = disk_usage.total / (1024**3)
                result["disk_usage_percent"] = (1 - disk_usage.free / disk_usage.total) * 100
                
                if result["disk_free_gb"] < 1.0:
                    result["status"] = "warning"
                    result["message"] = "Low disk space"
            except Exception:
                result["disk_info"] = "unavailable"
            
            # Check for missing directories
            missing_dirs = []
            for key, exists in result.items():
                if key.endswith("_exists") and not exists:
                    missing_dirs.append(key.replace("_exists", ""))
            
            if missing_dirs:
                result["status"] = "warning"
                result["message"] = f"Missing directories: {', '.join(missing_dirs)}"
            
            self.logger.info("File system check completed")
            return result
            
        except Exception as e:
            self.logger.error(f"File system check failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def check_network_connectivity(self) -> Dict[str, Any]:
        """Check network connectivity"""
        self.logger.info("Checking network connectivity...")
        
        try:
            result = {
                "status": "healthy",
                "tests": {}
            }
            
            # Test FRED API connectivity
            try:
                fred_response = requests.get(
                    "https://api.stlouisfed.org/fred/series?series_id=GDP&api_key=test&file_type=json",
                    timeout=10
                )
                result["tests"]["fred_api"] = {
                    "reachable": True,
                    "response_time": fred_response.elapsed.total_seconds(),
                    "status_code": fred_response.status_code
                }
            except Exception as e:
                result["tests"]["fred_api"] = {
                    "reachable": False,
                    "error": str(e)
                }
            
            # Test general internet connectivity
            try:
                google_response = requests.get("https://www.google.com", timeout=5)
                result["tests"]["internet"] = {
                    "reachable": True,
                    "response_time": google_response.elapsed.total_seconds()
                }
            except Exception as e:
                result["tests"]["internet"] = {
                    "reachable": False,
                    "error": str(e)
                }
                result["status"] = "error"
            
            # Test AWS connectivity (if configured)
            try:
                from config.settings import get_config
                config = get_config()
                if config.aws.access_key_id:
                    import boto3
                    sts = boto3.client('sts')
                    sts.get_caller_identity()
                    result["tests"]["aws"] = {
                        "reachable": True,
                        "authenticated": True
                    }
                else:
                    result["tests"]["aws"] = {
                        "reachable": "not_configured"
                    }
            except Exception as e:
                result["tests"]["aws"] = {
                    "reachable": False,
                    "error": str(e)
                }
            
            self.logger.info("Network connectivity check completed")
            return result
            
        except Exception as e:
            self.logger.error(f"Network connectivity check failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def check_application_modules(self) -> Dict[str, Any]:
        """Check application module health"""
        self.logger.info("Checking application modules...")
        
        try:
            result = {
                "status": "healthy",
                "modules": {}
            }
            
            # Test core module imports
            core_modules = [
                ("src.core.enhanced_fred_client", "EnhancedFREDClient"),
                ("src.analysis.comprehensive_analytics", "ComprehensiveAnalytics"),
                ("src.analysis.economic_forecasting", "EconomicForecaster"),
                ("src.analysis.economic_segmentation", "EconomicSegmentation"),
                ("src.analysis.statistical_modeling", "StatisticalModeling"),
                ("src.analysis.mathematical_fixes", "MathematicalFixes"),
            ]
            
            for module_name, class_name in core_modules:
                try:
                    module_obj = __import__(module_name, fromlist=[class_name])
                    class_obj = getattr(module_obj, class_name)
                    result["modules"][module_name] = {
                        "importable": True,
                        "class_available": True
                    }
                except ImportError as e:
                    result["modules"][module_name] = {
                        "importable": False,
                        "error": str(e)
                    }
                    result["status"] = "warning"
                except Exception as e:
                    result["modules"][module_name] = {
                        "importable": True,
                        "class_available": False,
                        "error": str(e)
                    }
                    result["status"] = "warning"
            
            self.logger.info("Application modules check completed")
            return result
            
        except Exception as e:
            self.logger.error(f"Application modules check failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def check_test_suite(self) -> Dict[str, Any]:
        """Check test suite health"""
        self.logger.info("Checking test suite...")
        
        try:
            result = {
                "status": "healthy",
                "test_files": {}
            }
            
            # Check test directory structure
            test_dirs = ["tests/unit", "tests/integration", "tests/e2e"]
            for test_dir in test_dirs:
                dir_path = self.project_root / test_dir
                if dir_path.exists():
                    test_files = list(dir_path.glob("test_*.py"))
                    result["test_files"][test_dir] = {
                        "exists": True,
                        "file_count": len(test_files),
                        "files": [f.name for f in test_files]
                    }
                else:
                    result["test_files"][test_dir] = {
                        "exists": False,
                        "file_count": 0,
                        "files": []
                    }
                    result["status"] = "warning"
            
            # Check test runner
            test_runner = self.project_root / "tests" / "run_tests.py"
            result["test_runner"] = {
                "exists": test_runner.exists(),
                "executable": test_runner.exists() and os.access(test_runner, os.X_OK)
            }
            
            if not result["test_runner"]["exists"]:
                result["status"] = "warning"
            
            self.logger.info("Test suite check completed")
            return result
            
        except Exception as e:
            self.logger.error(f"Test suite check failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def check_performance(self) -> Dict[str, Any]:
        """Check system performance"""
        self.logger.info("Checking system performance...")
        
        try:
            import psutil
            import time
            
            result = {
                "status": "healthy",
                "performance": {}
            }
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            result["performance"]["cpu_usage"] = cpu_percent
            
            # Memory usage
            memory = psutil.virtual_memory()
            result["performance"]["memory_usage"] = memory.percent
            result["performance"]["memory_available_gb"] = memory.available / (1024**3)
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            if disk_io:
                result["performance"]["disk_read_mb"] = disk_io.read_bytes / (1024**2)
                result["performance"]["disk_write_mb"] = disk_io.write_bytes / (1024**2)
            
            # Performance thresholds
            if cpu_percent > 80:
                result["status"] = "warning"
                result["message"] = "High CPU usage"
            
            if memory.percent > 80:
                result["status"] = "warning"
                result["message"] = "High memory usage"
            
            self.logger.info("Performance check completed")
            return result
            
        except ImportError:
            self.logger.warning("psutil not installed - performance monitoring disabled")
            return {
                "status": "warning",
                "message": "psutil not installed - install with: pip install psutil"
            }
        except Exception as e:
            self.logger.error(f"Performance check failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def run_all_checks(self) -> Dict[str, Any]:
        """Run all health checks"""
        self.logger.info("Starting comprehensive health check...")
        
        checks = [
            ("python_environment", self.check_python_environment),
            ("dependencies", self.check_dependencies),
            ("configuration", self.check_configuration),
            ("file_system", self.check_file_system),
            ("network_connectivity", self.check_network_connectivity),
            ("application_modules", self.check_application_modules),
            ("test_suite", self.check_test_suite),
            ("performance", self.check_performance),
        ]
        
        for check_name, check_func in checks:
            try:
                self.health_results[check_name] = check_func()
            except Exception as e:
                self.health_results[check_name] = {
                    "status": "error",
                    "error": str(e)
                }
        
        # Calculate overall health
        overall_status = self._calculate_overall_health()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "duration": time.time() - self.start_time,
            "overall_status": overall_status,
            "checks": self.health_results
        }
    
    def _calculate_overall_health(self) -> str:
        """Calculate overall system health"""
        statuses = [check.get("status", "unknown") for check in self.health_results.values()]
        
        if "error" in statuses:
            return "error"
        elif "warning" in statuses:
            return "warning"
        else:
            return "healthy"
    
    def print_health_report(self, health_report: Dict[str, Any]):
        """Print comprehensive health report"""
        print("\n" + "=" * 60)
        print("🏥 FRED ML - SYSTEM HEALTH REPORT")
        print("=" * 60)
        
        overall_status = health_report["overall_status"]
        duration = health_report["duration"]
        
        # Status indicator
        status_icons = {
            "healthy": "✅",
            "warning": "⚠️",
            "error": "❌"
        }
        
        print(f"\nOverall Status: {status_icons.get(overall_status, '❓')} {overall_status.upper()}")
        print(f"Check Duration: {duration:.2f} seconds")
        print(f"Timestamp: {health_report['timestamp']}")
        
        print(f"\n📊 Detailed Results:")
        for check_name, check_result in health_report["checks"].items():
            status = check_result.get("status", "unknown")
            icon = status_icons.get(status, "❓")
            print(f"  {icon} {check_name.replace('_', ' ').title()}: {status}")
            
            if "message" in check_result:
                print(f"    └─ {check_result['message']}")
        
        # Summary
        print(f"\n📈 Summary:")
        status_counts = {}
        for check_result in health_report["checks"].values():
            status = check_result.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
        
        for status, count in status_counts.items():
            icon = status_icons.get(status, "❓")
            print(f"  {icon} {status.title()}: {count} checks")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        if overall_status == "healthy":
            print("  ✅ System is healthy and ready for production use")
        elif overall_status == "warning":
            print("  ⚠️ System has some issues that should be addressed")
            for check_name, check_result in health_report["checks"].items():
                if check_result.get("status") == "warning":
                    print(f"    - Review {check_name.replace('_', ' ')} configuration")
        else:
            print("  ❌ System has critical issues that must be resolved")
            for check_name, check_result in health_report["checks"].items():
                if check_result.get("status") == "error":
                    print(f"    - Fix {check_name.replace('_', ' ')} issues")
    
    def save_health_report(self, health_report: Dict[str, Any], filename: str = "health_report.json"):
        """Save health report to file"""
        report_path = self.project_root / filename
        try:
            with open(report_path, 'w') as f:
                json.dump(health_report, f, indent=2, default=str)
            self.logger.info(f"Health report saved to: {report_path}")
        except Exception as e:
            self.logger.error(f"Failed to save health report: {e}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="FRED ML Health Checker")
    parser.add_argument("--save-report", action="store_true", help="Save health report to file")
    parser.add_argument("--output-file", default="health_report.json", help="Output file for health report")
    
    args = parser.parse_args()
    
    checker = HealthChecker()
    health_report = checker.run_all_checks()
    
    checker.print_health_report(health_report)
    
    if args.save_report:
        checker.save_health_report(health_report, args.output_file)
    
    # Exit with appropriate code
    if health_report["overall_status"] == "error":
        sys.exit(1)
    elif health_report["overall_status"] == "warning":
        sys.exit(2)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main() 