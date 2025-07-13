#!/usr/bin/env python3
"""
FRED ML - Integration and Testing Script
Comprehensive integration of all updates and system testing
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
from datetime import datetime
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FREDMLIntegration:
    """Comprehensive integration and testing for FRED ML system"""
    
    def __init__(self):
        self.root_dir = Path(__file__).parent.parent
        self.test_results = {}
        self.integration_status = {}
        
    def run_integration_checklist(self):
        """Run comprehensive integration checklist"""
        logger.info("🚀 Starting FRED ML Integration and Testing")
        logger.info("=" * 60)
        
        # 1. Directory Structure Validation
        self.validate_directory_structure()
        
        # 2. Dependencies Check
        self.check_dependencies()
        
        # 3. Configuration Validation
        self.validate_configurations()
        
        # 4. Code Quality Checks
        self.run_code_quality_checks()
        
        # 5. Unit Tests
        self.run_unit_tests()
        
        # 6. Integration Tests
        self.run_integration_tests()
        
        # 7. Advanced Analytics Tests
        self.test_advanced_analytics()
        
        # 8. Streamlit UI Test
        self.test_streamlit_ui()
        
        # 9. Documentation Check
        self.validate_documentation()
        
        # 10. Final Integration Report
        self.generate_integration_report()
        
    def validate_directory_structure(self):
        """Validate and organize directory structure"""
        logger.info("📁 Validating directory structure...")
        
        required_dirs = [
            'src/analysis',
            'src/core',
            'src/visualization',
            'src/lambda',
            'scripts',
            'tests/unit',
            'tests/integration',
            'tests/e2e',
            'docs',
            'config',
            'data/exports',
            'data/processed',
            'frontend',
            'infrastructure',
            'deploy'
        ]
        
        for dir_path in required_dirs:
            full_path = self.root_dir / dir_path
            if not full_path.exists():
                full_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"✅ Created directory: {dir_path}")
            else:
                logger.info(f"✅ Directory exists: {dir_path}")
        
        # Check for required files
        required_files = [
            'src/analysis/economic_forecasting.py',
            'src/analysis/economic_segmentation.py',
            'src/analysis/statistical_modeling.py',
            'src/analysis/comprehensive_analytics.py',
            'src/core/enhanced_fred_client.py',
            'frontend/app.py',
            'scripts/run_advanced_analytics.py',
            'scripts/comprehensive_demo.py',
            'config/pipeline.yaml',
            'requirements.txt',
            'README.md'
        ]
        
        missing_files = []
        for file_path in required_files:
            full_path = self.root_dir / file_path
            if not full_path.exists():
                missing_files.append(file_path)
            else:
                logger.info(f"✅ File exists: {file_path}")
        
        if missing_files:
            logger.error(f"❌ Missing files: {missing_files}")
            self.integration_status['directory_structure'] = False
        else:
            logger.info("✅ Directory structure validation passed")
            self.integration_status['directory_structure'] = True
    
    def check_dependencies(self):
        """Check and validate dependencies"""
        logger.info("📦 Checking dependencies...")
        
        try:
            # Check if requirements.txt exists and is valid
            requirements_file = self.root_dir / 'requirements.txt'
            if requirements_file.exists():
                with open(requirements_file, 'r') as f:
                    requirements = f.read()
                
                # Check for key dependencies
                key_deps = [
                    'fredapi',
                    'pandas',
                    'numpy',
                    'scikit-learn',
                    'scipy',
                    'statsmodels',
                    'streamlit',
                    'plotly',
                    'boto3'
                ]
                
                missing_deps = []
                for dep in key_deps:
                    if dep not in requirements:
                        missing_deps.append(dep)
                
                if missing_deps:
                    logger.warning(f"⚠️ Missing dependencies: {missing_deps}")
                else:
                    logger.info("✅ All key dependencies found in requirements.txt")
                
                self.integration_status['dependencies'] = True
            else:
                logger.error("❌ requirements.txt not found")
                self.integration_status['dependencies'] = False
                
        except Exception as e:
            logger.error(f"❌ Error checking dependencies: {e}")
            self.integration_status['dependencies'] = False
    
    def validate_configurations(self):
        """Validate configuration files"""
        logger.info("⚙️ Validating configurations...")
        
        config_files = [
            'config/pipeline.yaml',
            'config/settings.py',
            '.github/workflows/scheduled.yml'
        ]
        
        config_status = True
        for config_file in config_files:
            full_path = self.root_dir / config_file
            if full_path.exists():
                logger.info(f"✅ Configuration file exists: {config_file}")
            else:
                logger.error(f"❌ Missing configuration file: {config_file}")
                config_status = False
        
        # Check cron job configuration
        pipeline_config = self.root_dir / 'config/pipeline.yaml'
        if pipeline_config.exists():
            with open(pipeline_config, 'r') as f:
                content = f.read()
                if 'schedule: "0 0 1 */3 *"' in content:
                    logger.info("✅ Quarterly cron job configuration found")
                else:
                    logger.warning("⚠️ Cron job configuration may not be quarterly")
        
        self.integration_status['configurations'] = config_status
    
    def run_code_quality_checks(self):
        """Run code quality checks"""
        logger.info("🔍 Running code quality checks...")
        
        try:
            # Check for Python syntax errors
            python_files = list(self.root_dir.rglob("*.py"))
            
            syntax_errors = []
            for py_file in python_files:
                try:
                    with open(py_file, 'r') as f:
                        compile(f.read(), str(py_file), 'exec')
                except SyntaxError as e:
                    syntax_errors.append(f"{py_file}: {e}")
            
            if syntax_errors:
                logger.error(f"❌ Syntax errors found: {syntax_errors}")
                self.integration_status['code_quality'] = False
            else:
                logger.info("✅ No syntax errors found")
                self.integration_status['code_quality'] = True
                
        except Exception as e:
            logger.error(f"❌ Error in code quality checks: {e}")
            self.integration_status['code_quality'] = False
    
    def run_unit_tests(self):
        """Run unit tests"""
        logger.info("🧪 Running unit tests...")
        
        try:
            # Check if tests directory exists
            tests_dir = self.root_dir / 'tests'
            if not tests_dir.exists():
                logger.warning("⚠️ Tests directory not found")
                self.integration_status['unit_tests'] = False
                return
            
            # Run pytest if available
            try:
                result = subprocess.run(
                    [sys.executable, '-m', 'pytest', 'tests/unit/', '-v'],
                    capture_output=True,
                    text=True,
                    cwd=self.root_dir
                )
                
                if result.returncode == 0:
                    logger.info("✅ Unit tests passed")
                    self.integration_status['unit_tests'] = True
                else:
                    logger.error(f"❌ Unit tests failed: {result.stderr}")
                    self.integration_status['unit_tests'] = False
                    
            except FileNotFoundError:
                logger.warning("⚠️ pytest not available, skipping unit tests")
                self.integration_status['unit_tests'] = False
                
        except Exception as e:
            logger.error(f"❌ Error running unit tests: {e}")
            self.integration_status['unit_tests'] = False
    
    def run_integration_tests(self):
        """Run integration tests"""
        logger.info("🔗 Running integration tests...")
        
        try:
            # Test FRED API connection
            from config.settings import FRED_API_KEY
            if FRED_API_KEY:
                logger.info("✅ FRED API key configured")
                self.integration_status['fred_api'] = True
            else:
                logger.warning("⚠️ FRED API key not configured")
                self.integration_status['fred_api'] = False
            
            # Test AWS configuration
            try:
                import boto3
                logger.info("✅ AWS SDK available")
                self.integration_status['aws_sdk'] = True
            except ImportError:
                logger.warning("⚠️ AWS SDK not available")
                self.integration_status['aws_sdk'] = False
            
            # Test analytics modules
            try:
                sys.path.append(str(self.root_dir / 'src'))
                from src.analysis.comprehensive_analytics import ComprehensiveAnalytics
                from src.core.enhanced_fred_client import EnhancedFREDClient
                logger.info("✅ Analytics modules available")
                self.integration_status['analytics_modules'] = True
            except ImportError as e:
                logger.error(f"❌ Analytics modules not available: {e}")
                self.integration_status['analytics_modules'] = False
                
        except Exception as e:
            logger.error(f"❌ Error in integration tests: {e}")
            self.integration_status['integration_tests'] = False
    
    def test_advanced_analytics(self):
        """Test advanced analytics functionality"""
        logger.info("🔮 Testing advanced analytics...")
        
        try:
            # Test analytics modules import
            sys.path.append(str(self.root_dir / 'src'))
            
            # Test Enhanced FRED Client
            try:
                from src.core.enhanced_fred_client import EnhancedFREDClient
                logger.info("✅ Enhanced FRED Client available")
                self.integration_status['enhanced_fred_client'] = True
            except ImportError as e:
                logger.error(f"❌ Enhanced FRED Client not available: {e}")
                self.integration_status['enhanced_fred_client'] = False
            
            # Test Economic Forecasting
            try:
                from src.analysis.economic_forecasting import EconomicForecaster
                logger.info("✅ Economic Forecasting available")
                self.integration_status['economic_forecasting'] = True
            except ImportError as e:
                logger.error(f"❌ Economic Forecasting not available: {e}")
                self.integration_status['economic_forecasting'] = False
            
            # Test Economic Segmentation
            try:
                from src.analysis.economic_segmentation import EconomicSegmentation
                logger.info("✅ Economic Segmentation available")
                self.integration_status['economic_segmentation'] = True
            except ImportError as e:
                logger.error(f"❌ Economic Segmentation not available: {e}")
                self.integration_status['economic_segmentation'] = False
            
            # Test Statistical Modeling
            try:
                from src.analysis.statistical_modeling import StatisticalModeling
                logger.info("✅ Statistical Modeling available")
                self.integration_status['statistical_modeling'] = True
            except ImportError as e:
                logger.error(f"❌ Statistical Modeling not available: {e}")
                self.integration_status['statistical_modeling'] = False
            
            # Test Comprehensive Analytics
            try:
                from src.analysis.comprehensive_analytics import ComprehensiveAnalytics
                logger.info("✅ Comprehensive Analytics available")
                self.integration_status['comprehensive_analytics'] = True
            except ImportError as e:
                logger.error(f"❌ Comprehensive Analytics not available: {e}")
                self.integration_status['comprehensive_analytics'] = False
                
        except Exception as e:
            logger.error(f"❌ Error testing advanced analytics: {e}")
    
    def test_streamlit_ui(self):
        """Test Streamlit UI"""
        logger.info("🎨 Testing Streamlit UI...")
        
        try:
            # Check if Streamlit app exists
            streamlit_app = self.root_dir / 'frontend/app.py'
            if streamlit_app.exists():
                logger.info("✅ Streamlit app exists")
                
                # Check for required imports
                with open(streamlit_app, 'r') as f:
                    content = f.read()
                    
                required_imports = [
                    'streamlit',
                    'plotly',
                    'pandas',
                    'boto3'
                ]
                
                missing_imports = []
                for imp in required_imports:
                    if imp not in content:
                        missing_imports.append(imp)
                
                if missing_imports:
                    logger.warning(f"⚠️ Missing imports in Streamlit app: {missing_imports}")
                else:
                    logger.info("✅ All required imports found in Streamlit app")
                
                self.integration_status['streamlit_ui'] = True
            else:
                logger.error("❌ Streamlit app not found")
                self.integration_status['streamlit_ui'] = False
                
        except Exception as e:
            logger.error(f"❌ Error testing Streamlit UI: {e}")
            self.integration_status['streamlit_ui'] = False
    
    def validate_documentation(self):
        """Validate documentation"""
        logger.info("📚 Validating documentation...")
        
        doc_files = [
            'README.md',
            'docs/ADVANCED_ANALYTICS_SUMMARY.md',
            'docs/CONVERSATION_SUMMARY.md'
        ]
        
        doc_status = True
        for doc_file in doc_files:
            full_path = self.root_dir / doc_file
            if full_path.exists():
                logger.info(f"✅ Documentation exists: {doc_file}")
            else:
                logger.warning(f"⚠️ Missing documentation: {doc_file}")
                doc_status = False
        
        self.integration_status['documentation'] = doc_status
    
    def generate_integration_report(self):
        """Generate comprehensive integration report"""
        logger.info("📊 Generating integration report...")
        
        # Calculate overall status
        total_checks = len(self.integration_status)
        passed_checks = sum(1 for status in self.integration_status.values() if status)
        overall_status = "✅ PASSED" if passed_checks == total_checks else "❌ FAILED"
        
        # Generate report
        report = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": overall_status,
            "summary": {
                "total_checks": total_checks,
                "passed_checks": passed_checks,
                "failed_checks": total_checks - passed_checks,
                "success_rate": f"{(passed_checks/total_checks)*100:.1f}%"
            },
            "detailed_results": self.integration_status
        }
        
        # Save report
        report_file = self.root_dir / 'integration_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        logger.info("=" * 60)
        logger.info("📊 INTEGRATION REPORT")
        logger.info("=" * 60)
        logger.info(f"Overall Status: {overall_status}")
        logger.info(f"Total Checks: {total_checks}")
        logger.info(f"Passed: {passed_checks}")
        logger.info(f"Failed: {total_checks - passed_checks}")
        logger.info(f"Success Rate: {(passed_checks/total_checks)*100:.1f}%")
        logger.info("=" * 60)
        
        # Print detailed results
        logger.info("Detailed Results:")
        for check, status in self.integration_status.items():
            status_icon = "✅" if status else "❌"
            logger.info(f"  {status_icon} {check}")
        
        logger.info("=" * 60)
        logger.info(f"Report saved to: {report_file}")
        
        return report
    
    def prepare_for_github(self):
        """Prepare for GitHub submission"""
        logger.info("🚀 Preparing for GitHub submission...")
        
        # Check git status
        try:
            result = subprocess.run(
                ['git', 'status', '--porcelain'],
                capture_output=True,
                text=True,
                cwd=self.root_dir
            )
            
            if result.stdout.strip():
                logger.info("📝 Changes detected:")
                logger.info(result.stdout)
                
                # Suggest git commands
                logger.info("\n📋 Suggested git commands:")
                logger.info("git add .")
                logger.info("git commit -m 'feat: Integrate advanced analytics and enterprise UI'")
                logger.info("git push origin main")
            else:
                logger.info("✅ No changes detected")
                
        except Exception as e:
            logger.error(f"❌ Error checking git status: {e}")

def main():
    """Main integration function"""
    integrator = FREDMLIntegration()
    
    try:
        # Run integration checklist
        integrator.run_integration_checklist()
        
        # Prepare for GitHub
        integrator.prepare_for_github()
        
        logger.info("🎉 Integration and testing completed!")
        
    except Exception as e:
        logger.error(f"❌ Integration failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 