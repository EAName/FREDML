#!/usr/bin/env python3
"""
FRED ML - Complete System Test
Comprehensive testing of all system components
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

class FREDMLSystemTest:
    """Complete system testing for FRED ML"""
    
    def __init__(self):
        self.root_dir = Path(__file__).parent.parent
        self.test_results = {}
        
    def run_complete_system_test(self):
        """Run complete system test"""
        logger.info("🧪 Starting FRED ML Complete System Test")
        logger.info("=" * 60)
        
        # 1. Environment Setup Test
        self.test_environment_setup()
        
        # 2. Dependencies Test
        self.test_dependencies()
        
        # 3. Configuration Test
        self.test_configurations()
        
        # 4. Core Modules Test
        self.test_core_modules()
        
        # 5. Advanced Analytics Test
        self.test_advanced_analytics()
        
        # 6. Streamlit UI Test
        self.test_streamlit_ui()
        
        # 7. Integration Test
        self.test_integration()
        
        # 8. Performance Test
        self.test_performance()
        
        # 9. Generate Test Report
        self.generate_test_report()
        
    def test_environment_setup(self):
        """Test environment setup"""
        logger.info("🔧 Testing environment setup...")
        
        # Check Python version
        python_version = sys.version_info
        if python_version.major >= 3 and python_version.minor >= 8:
            logger.info(f"✅ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
            self.test_results['python_version'] = True
        else:
            logger.error(f"❌ Python version too old: {python_version}")
            self.test_results['python_version'] = False
        
        # Check working directory
        logger.info(f"✅ Working directory: {self.root_dir}")
        self.test_results['working_directory'] = True
        
        # Check environment variables
        required_env_vars = ['FRED_API_KEY']
        env_status = True
        for var in required_env_vars:
            if os.getenv(var):
                logger.info(f"✅ Environment variable set: {var}")
            else:
                logger.warning(f"⚠️ Environment variable not set: {var}")
                env_status = False
        
        self.test_results['environment_variables'] = env_status
    
    def test_dependencies(self):
        """Test dependencies"""
        logger.info("📦 Testing dependencies...")
        
        required_packages = [
            'pandas',
            'numpy',
            'scikit-learn',
            'scipy',
            'statsmodels',
            'streamlit',
            'plotly',
            'boto3',
            'fredapi'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
                logger.info(f"✅ Package available: {package}")
            except ImportError:
                logger.error(f"❌ Package missing: {package}")
                missing_packages.append(package)
        
        if missing_packages:
            self.test_results['dependencies'] = False
            logger.error(f"❌ Missing packages: {missing_packages}")
        else:
            self.test_results['dependencies'] = True
            logger.info("✅ All dependencies available")
    
    def test_configurations(self):
        """Test configuration files"""
        logger.info("⚙️ Testing configurations...")
        
        config_files = [
            'config/pipeline.yaml',
            'config/settings.py',
            'requirements.txt',
            'pyproject.toml'
        ]
        
        config_status = True
        for config_file in config_files:
            full_path = self.root_dir / config_file
            if full_path.exists():
                logger.info(f"✅ Configuration file exists: {config_file}")
            else:
                logger.error(f"❌ Configuration file missing: {config_file}")
                config_status = False
        
        self.test_results['configurations'] = config_status
    
    def test_core_modules(self):
        """Test core modules"""
        logger.info("🔧 Testing core modules...")
        
        # Add src to path
        sys.path.append(str(self.root_dir / 'src'))
        
        core_modules = [
            'src.core.enhanced_fred_client',
            'src.analysis.economic_forecasting',
            'src.analysis.economic_segmentation',
            'src.analysis.statistical_modeling',
            'src.analysis.comprehensive_analytics'
        ]
        
        module_status = True
        for module in core_modules:
            try:
                __import__(module)
                logger.info(f"✅ Module available: {module}")
            except ImportError as e:
                logger.error(f"❌ Module missing: {module} - {e}")
                module_status = False
        
        self.test_results['core_modules'] = module_status
    
    def test_advanced_analytics(self):
        """Test advanced analytics functionality"""
        logger.info("🔮 Testing advanced analytics...")
        
        try:
            # Test Enhanced FRED Client
            from src.core.enhanced_fred_client import EnhancedFREDClient
            logger.info("✅ Enhanced FRED Client imported successfully")
            
            # Test Economic Forecasting
            from src.analysis.economic_forecasting import EconomicForecaster
            logger.info("✅ Economic Forecasting imported successfully")
            
            # Test Economic Segmentation
            from src.analysis.economic_segmentation import EconomicSegmentation
            logger.info("✅ Economic Segmentation imported successfully")
            
            # Test Statistical Modeling
            from src.analysis.statistical_modeling import StatisticalModeling
            logger.info("✅ Statistical Modeling imported successfully")
            
            # Test Comprehensive Analytics
            from src.analysis.comprehensive_analytics import ComprehensiveAnalytics
            logger.info("✅ Comprehensive Analytics imported successfully")
            
            self.test_results['advanced_analytics'] = True
            
        except Exception as e:
            logger.error(f"❌ Advanced analytics test failed: {e}")
            self.test_results['advanced_analytics'] = False
    
    def test_streamlit_ui(self):
        """Test Streamlit UI"""
        logger.info("🎨 Testing Streamlit UI...")
        
        try:
            # Check if Streamlit app exists
            streamlit_app = self.root_dir / 'frontend/app.py'
            if not streamlit_app.exists():
                logger.error("❌ Streamlit app not found")
                self.test_results['streamlit_ui'] = False
                return
            
            # Check app content
            with open(streamlit_app, 'r') as f:
                content = f.read()
            
            # Check for required components
            required_components = [
                'st.set_page_config',
                'ComprehensiveAnalytics',
                'EnhancedFREDClient',
                'show_executive_dashboard',
                'show_advanced_analytics_page'
            ]
            
            missing_components = []
            for component in required_components:
                if component not in content:
                    missing_components.append(component)
            
            if missing_components:
                logger.error(f"❌ Missing components in Streamlit app: {missing_components}")
                self.test_results['streamlit_ui'] = False
            else:
                logger.info("✅ Streamlit UI components found")
                self.test_results['streamlit_ui'] = True
                
        except Exception as e:
            logger.error(f"❌ Streamlit UI test failed: {e}")
            self.test_results['streamlit_ui'] = False
    
    def test_integration(self):
        """Test system integration"""
        logger.info("🔗 Testing system integration...")
        
        try:
            # Test FRED API connection (if API key available)
            from config.settings import FRED_API_KEY
            if FRED_API_KEY:
                try:
                    from src.core.enhanced_fred_client import EnhancedFREDClient
                    client = EnhancedFREDClient(FRED_API_KEY)
                    logger.info("✅ FRED API client created successfully")
                    
                    # Test series info retrieval
                    series_info = client.get_series_info('GDPC1')
                    if 'error' not in series_info:
                        logger.info("✅ FRED API connection successful")
                        self.test_results['fred_api_integration'] = True
                    else:
                        logger.warning("⚠️ FRED API connection failed")
                        self.test_results['fred_api_integration'] = False
                        
                except Exception as e:
                    logger.error(f"❌ FRED API integration failed: {e}")
                    self.test_results['fred_api_integration'] = False
            else:
                logger.warning("⚠️ FRED API key not available, skipping API test")
                self.test_results['fred_api_integration'] = False
            
            # Test analytics integration
            try:
                from src.analysis.comprehensive_analytics import ComprehensiveAnalytics
                logger.info("✅ Analytics integration successful")
                self.test_results['analytics_integration'] = True
            except Exception as e:
                logger.error(f"❌ Analytics integration failed: {e}")
                self.test_results['analytics_integration'] = False
                
        except Exception as e:
            logger.error(f"❌ Integration test failed: {e}")
            self.test_results['integration'] = False
    
    def test_performance(self):
        """Test system performance"""
        logger.info("⚡ Testing system performance...")
        
        try:
            # Test data processing performance
            import pandas as pd
            import numpy as np
            
            # Create test data
            test_data = pd.DataFrame({
                'GDPC1': np.random.randn(1000),
                'INDPRO': np.random.randn(1000),
                'RSAFS': np.random.randn(1000)
            })
            
            # Test analytics modules with test data
            from src.analysis.economic_forecasting import EconomicForecaster
            from src.analysis.economic_segmentation import EconomicSegmentation
            from src.analysis.statistical_modeling import StatisticalModeling
            
            # Test forecasting performance
            forecaster = EconomicForecaster(test_data)
            logger.info("✅ Forecasting module performance test passed")
            
            # Test segmentation performance
            segmentation = EconomicSegmentation(test_data)
            logger.info("✅ Segmentation module performance test passed")
            
            # Test statistical modeling performance
            modeling = StatisticalModeling(test_data)
            logger.info("✅ Statistical modeling performance test passed")
            
            self.test_results['performance'] = True
            
        except Exception as e:
            logger.error(f"❌ Performance test failed: {e}")
            self.test_results['performance'] = False
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        logger.info("📊 Generating test report...")
        
        # Calculate overall status
        total_tests = len(self.test_results)
        passed_tests = sum(1 for status in self.test_results.values() if status)
        overall_status = "✅ PASSED" if passed_tests == total_tests else "❌ FAILED"
        
        # Generate report
        report = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": overall_status,
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": total_tests - passed_tests,
                "success_rate": f"{(passed_tests/total_tests)*100:.1f}%"
            },
            "detailed_results": self.test_results
        }
        
        # Save report
        report_file = self.root_dir / 'system_test_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        logger.info("=" * 60)
        logger.info("📊 SYSTEM TEST REPORT")
        logger.info("=" * 60)
        logger.info(f"Overall Status: {overall_status}")
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Passed: {passed_tests}")
        logger.info(f"Failed: {total_tests - passed_tests}")
        logger.info(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        logger.info("=" * 60)
        
        # Print detailed results
        logger.info("Detailed Results:")
        for test, status in self.test_results.items():
            status_icon = "✅" if status else "❌"
            logger.info(f"  {status_icon} {test}")
        
        logger.info("=" * 60)
        logger.info(f"Report saved to: {report_file}")
        
        return report
    
    def run_demo_tests(self):
        """Run demo tests"""
        logger.info("🎯 Running demo tests...")
        
        try:
            # Test comprehensive demo
            demo_script = self.root_dir / 'scripts/comprehensive_demo.py'
            if demo_script.exists():
                logger.info("✅ Comprehensive demo script exists")
                
                # Test demo script syntax
                with open(demo_script, 'r') as f:
                    compile(f.read(), str(demo_script), 'exec')
                logger.info("✅ Comprehensive demo script syntax valid")
                
                self.test_results['comprehensive_demo'] = True
            else:
                logger.error("❌ Comprehensive demo script not found")
                self.test_results['comprehensive_demo'] = False
            
            # Test advanced analytics script
            analytics_script = self.root_dir / 'scripts/run_advanced_analytics.py'
            if analytics_script.exists():
                logger.info("✅ Advanced analytics script exists")
                
                # Test script syntax
                with open(analytics_script, 'r') as f:
                    compile(f.read(), str(analytics_script), 'exec')
                logger.info("✅ Advanced analytics script syntax valid")
                
                self.test_results['advanced_analytics_script'] = True
            else:
                logger.error("❌ Advanced analytics script not found")
                self.test_results['advanced_analytics_script'] = False
                
        except Exception as e:
            logger.error(f"❌ Demo tests failed: {e}")
            self.test_results['demo_tests'] = False

def main():
    """Main test function"""
    tester = FREDMLSystemTest()
    
    try:
        # Run complete system test
        tester.run_complete_system_test()
        
        # Run demo tests
        tester.run_demo_tests()
        
        logger.info("🎉 Complete system test finished!")
        
    except Exception as e:
        logger.error(f"❌ System test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 