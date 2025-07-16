#!/usr/bin/env python3
"""
Enterprise-grade test runner for FRED ML
Consolidates all testing functionality into a single, comprehensive test suite
"""

import sys
import os
import subprocess
import argparse
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestRunner:
    """Enterprise-grade test runner for FRED ML"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.test_results = {}
        self.start_time = time.time()
        
    def run_command(self, command: List[str], capture_output: bool = True) -> Tuple[int, str, str]:
        """Run a command and return results"""
        try:
            result = subprocess.run(
                command,
                capture_output=capture_output,
                text=True,
                cwd=self.project_root
            )
            return result.returncode, result.stdout, result.stderr
        except Exception as e:
            return 1, "", str(e)
    
    def run_unit_tests(self) -> Dict:
        """Run unit tests"""
        print("🧪 Running Unit Tests...")
        print("=" * 50)
        
        start_time = time.time()
        returncode, stdout, stderr = self.run_command([
            "python", "-m", "pytest", "tests/unit/", "-v", "--tb=short"
        ])
        end_time = time.time()
        
        result = {
            "success": returncode == 0,
            "returncode": returncode,
            "stdout": stdout,
            "stderr": stderr,
            "duration": end_time - start_time
        }
        
        if result["success"]:
            print("✅ Unit tests passed")
        else:
            print("❌ Unit tests failed")
            if stderr:
                print(f"Error: {stderr}")
        
        return result
    
    def run_integration_tests(self) -> Dict:
        """Run integration tests"""
        print("\n🔗 Running Integration Tests...")
        print("=" * 50)
        
        start_time = time.time()
        returncode, stdout, stderr = self.run_command([
            "python", "-m", "pytest", "tests/integration/", "-v", "--tb=short"
        ])
        end_time = time.time()
        
        result = {
            "success": returncode == 0,
            "returncode": returncode,
            "stdout": stdout,
            "stderr": stderr,
            "duration": end_time - start_time
        }
        
        if result["success"]:
            print("✅ Integration tests passed")
        else:
            print("❌ Integration tests failed")
            if stderr:
                print(f"Error: {stderr}")
        
        return result
    
    def run_e2e_tests(self) -> Dict:
        """Run end-to-end tests"""
        print("\n🚀 Running End-to-End Tests...")
        print("=" * 50)
        
        start_time = time.time()
        returncode, stdout, stderr = self.run_command([
            "python", "-m", "pytest", "tests/e2e/", "-v", "--tb=short"
        ])
        end_time = time.time()
        
        result = {
            "success": returncode == 0,
            "returncode": returncode,
            "stdout": stdout,
            "stderr": stderr,
            "duration": end_time - start_time
        }
        
        if result["success"]:
            print("✅ End-to-end tests passed")
        else:
            print("❌ End-to-end tests failed")
            if stderr:
                print(f"Error: {stderr}")
        
        return result
    
    def run_import_tests(self) -> Dict:
        """Test module imports"""
        print("\n📦 Testing Module Imports...")
        print("=" * 50)
        
        start_time = time.time()
        
        # Test core imports
        import_tests = [
            ("src.core.enhanced_fred_client", "EnhancedFREDClient"),
            ("src.analysis.comprehensive_analytics", "ComprehensiveAnalytics"),
            ("src.analysis.economic_forecasting", "EconomicForecaster"),
            ("src.analysis.economic_segmentation", "EconomicSegmentation"),
            ("src.analysis.statistical_modeling", "StatisticalModeling"),
            ("src.analysis.mathematical_fixes", "MathematicalFixes"),
        ]
        
        failed_imports = []
        successful_imports = []
        
        for module, class_name in import_tests:
            try:
                module_obj = __import__(module, fromlist=[class_name])
                class_obj = getattr(module_obj, class_name)
                successful_imports.append(f"{module}.{class_name}")
                print(f"✅ {module}.{class_name}")
            except ImportError as e:
                failed_imports.append(f"{module}.{class_name} ({str(e)})")
                print(f"❌ {module}.{class_name} - {e}")
            except Exception as e:
                failed_imports.append(f"{module}.{class_name} ({str(e)})")
                print(f"❌ {module}.{class_name} - {e}")
        
        end_time = time.time()
        
        result = {
            "success": len(failed_imports) == 0,
            "successful_imports": successful_imports,
            "failed_imports": failed_imports,
            "duration": end_time - start_time
        }
        
        if result["success"]:
            print("✅ All imports successful")
        else:
            print(f"❌ {len(failed_imports)} imports failed")
        
        return result
    
    def run_code_quality_tests(self) -> Dict:
        """Run code quality checks"""
        print("\n🔍 Running Code Quality Checks...")
        print("=" * 50)
        
        start_time = time.time()
        
        # Check for common issues
        issues = []
        
        # Check for debug statements
        debug_files = []
        for root, dirs, files in os.walk(self.project_root):
            if "tests" in root or ".git" in root or "__pycache__" in root:
                continue
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r') as f:
                            content = f.read()
                            if 'print(' in content and 'DEBUG:' in content:
                                debug_files.append(file_path)
                    except Exception:
                        pass
        
        if debug_files:
            issues.append(f"Debug statements found in {len(debug_files)} files")
        
        # Check for TODO comments
        todo_files = []
        for root, dirs, files in os.walk(self.project_root):
            if "tests" in root or ".git" in root or "__pycache__" in root:
                continue
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r') as f:
                            content = f.read()
                            if 'TODO:' in content or 'FIXME:' in content:
                                todo_files.append(file_path)
                    except Exception:
                        pass
        
        if todo_files:
            issues.append(f"TODO/FIXME comments found in {len(todo_files)} files")
        
        end_time = time.time()
        
        result = {
            "success": len(issues) == 0,
            "issues": issues,
            "debug_files": debug_files,
            "todo_files": todo_files,
            "duration": end_time - start_time
        }
        
        if result["success"]:
            print("✅ Code quality checks passed")
        else:
            print("⚠️ Code quality issues found:")
            for issue in issues:
                print(f"  - {issue}")
        
        return result
    
    def run_performance_tests(self) -> Dict:
        """Run performance tests"""
        print("\n⚡ Running Performance Tests...")
        print("=" * 50)
        
        start_time = time.time()
        
        # Basic performance tests
        import time
        import pandas as pd
        import numpy as np
        
        performance_results = {}
        
        # Test data processing performance
        data_start = time.time()
        large_data = pd.DataFrame({
            'GDPC1': np.random.normal(22000, 1000, 1000),
            'INDPRO': np.random.normal(100, 5, 1000),
            'CPIAUCSL': np.random.normal(250, 10, 1000)
        })
        data_end = time.time()
        performance_results['data_creation'] = data_end - data_start
        
        # Test computation performance
        comp_start = time.time()
        large_data.describe()
        large_data.corr()
        comp_end = time.time()
        performance_results['computation'] = comp_end - comp_start
        
        end_time = time.time()
        
        result = {
            "success": performance_results['computation'] < 5.0,  # Should complete within 5 seconds
            "performance_results": performance_results,
            "duration": end_time - start_time
        }
        
        if result["success"]:
            print("✅ Performance tests passed")
            print(f"  - Data creation: {performance_results['data_creation']:.3f}s")
            print(f"  - Computation: {performance_results['computation']:.3f}s")
        else:
            print("❌ Performance tests failed - computation too slow")
        
        return result
    
    def generate_report(self) -> Dict:
        """Generate comprehensive test report"""
        total_duration = time.time() - self.start_time
        
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_duration": total_duration,
            "results": self.test_results,
            "summary": {
                "total_tests": len(self.test_results),
                "passed_tests": sum(1 for r in self.test_results.values() if r.get("success", False)),
                "failed_tests": sum(1 for r in self.test_results.values() if not r.get("success", True))
            }
        }
        
        return report
    
    def print_summary(self, report: Dict):
        """Print test summary"""
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        summary = report["summary"]
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed_tests']}")
        print(f"Failed: {summary['failed_tests']}")
        print(f"Success Rate: {(summary['passed_tests'] / summary['total_tests'] * 100):.1f}%")
        print(f"Total Duration: {report['total_duration']:.2f}s")
        
        print("\n📋 Detailed Results:")
        for test_name, result in report["results"].items():
            status = "✅ PASS" if result.get("success", False) else "❌ FAIL"
            duration = f"{result.get('duration', 0):.2f}s"
            print(f"  {test_name}: {status} ({duration})")
        
        if summary['failed_tests'] == 0:
            print("\n🎉 All tests passed! The system is ready for production.")
        else:
            print(f"\n⚠️ {summary['failed_tests']} test(s) failed. Please review the results above.")
    
    def save_report(self, report: Dict, filename: str = "test_report.json"):
        """Save test report to file"""
        report_path = self.project_root / filename
        try:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"\n📄 Test report saved to: {report_path}")
        except Exception as e:
            print(f"\n❌ Failed to save test report: {e}")
    
    def run_all_tests(self, save_report: bool = True) -> Dict:
        """Run all tests and generate report"""
        print("🚀 FRED ML - Enterprise Test Suite")
        print("=" * 60)
        
        # Run all test categories
        self.test_results["unit_tests"] = self.run_unit_tests()
        self.test_results["integration_tests"] = self.run_integration_tests()
        self.test_results["e2e_tests"] = self.run_e2e_tests()
        self.test_results["import_tests"] = self.run_import_tests()
        self.test_results["code_quality"] = self.run_code_quality_tests()
        self.test_results["performance_tests"] = self.run_performance_tests()
        
        # Generate and display report
        report = self.generate_report()
        self.print_summary(report)
        
        if save_report:
            self.save_report(report)
        
        return report


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="FRED ML Test Runner")
    parser.add_argument("--no-report", action="store_true", help="Don't save test report")
    parser.add_argument("--unit-only", action="store_true", help="Run only unit tests")
    parser.add_argument("--integration-only", action="store_true", help="Run only integration tests")
    parser.add_argument("--e2e-only", action="store_true", help="Run only end-to-end tests")
    
    args = parser.parse_args()
    
    runner = TestRunner()
    
    if args.unit_only:
        runner.test_results["unit_tests"] = runner.run_unit_tests()
    elif args.integration_only:
        runner.test_results["integration_tests"] = runner.run_integration_tests()
    elif args.e2e_only:
        runner.test_results["e2e_tests"] = runner.run_e2e_tests()
    else:
        runner.run_all_tests(save_report=not args.no_report)
    
    # Exit with appropriate code
    total_failed = sum(1 for r in runner.test_results.values() if not r.get("success", True))
    sys.exit(1 if total_failed > 0 else 0)


if __name__ == "__main__":
    main() 