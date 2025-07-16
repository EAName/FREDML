#!/usr/bin/env python3
"""
Enterprise-grade cleanup script for FRED ML
Identifies and removes redundant files to improve project organization
"""

import os
import shutil
import sys
from pathlib import Path
from typing import List, Dict, Set
import argparse


class ProjectCleaner:
    """Enterprise-grade project cleanup utility"""
    
    def __init__(self, dry_run: bool = True):
        self.project_root = Path(__file__).parent.parent
        self.dry_run = dry_run
        self.redundant_files = []
        self.removed_files = []
        self.kept_files = []
        
    def identify_redundant_test_files(self) -> List[Path]:
        """Identify redundant test files in root directory"""
        redundant_files = []
        
        # Files to be removed (redundant test files)
        redundant_patterns = [
            "test_analytics.py",
            "test_analytics_fix.py", 
            "test_real_analytics.py",
            "test_mathematical_fixes.py",
            "test_mathematical_fixes_fixed.py",
            "test_app.py",
            "test_local_app.py",
            "test_enhanced_app.py",
            "test_app_features.py",
            "test_frontend_data.py",
            "test_data_accuracy.py",
            "test_fred_frequency_issue.py",
            "test_imports.py",
            "test_gdp_scale.py",
            "test_data_validation.py",
            "test_alignment_divergence.py",
            "test_fixes_demonstration.py",
            "test_dynamic_scoring.py",
            "test_real_data_analysis.py",
            "test_math_issues.py",
            "simple_local_test.py",
            "debug_analytics.py",
            "debug_data_structure.py",
            "check_deployment.py"
        ]
        
        for pattern in redundant_patterns:
            file_path = self.project_root / pattern
            if file_path.exists():
                redundant_files.append(file_path)
                print(f"🔍 Found redundant file: {pattern}")
        
        return redundant_files
    
    def identify_debug_files(self) -> List[Path]:
        """Identify debug and temporary files"""
        debug_files = []
        
        # Debug and temporary files
        debug_patterns = [
            "alignment_divergence_insights.txt",
            "MATH_ISSUES_ANALYSIS.md",
            "test_report.json"
        ]
        
        for pattern in debug_patterns:
            file_path = self.project_root / pattern
            if file_path.exists():
                debug_files.append(file_path)
                print(f"🔍 Found debug file: {pattern}")
        
        return debug_files
    
    def identify_cache_directories(self) -> List[Path]:
        """Identify cache and temporary directories"""
        cache_dirs = []
        
        # Cache directories
        cache_patterns = [
            "__pycache__",
            ".pytest_cache",
            "htmlcov",
            "logs",
            "test_output"
        ]
        
        for pattern in cache_patterns:
            dir_path = self.project_root / pattern
            if dir_path.exists() and dir_path.is_dir():
                cache_dirs.append(dir_path)
                print(f"🔍 Found cache directory: {pattern}")
        
        return cache_dirs
    
    def backup_file(self, file_path: Path) -> Path:
        """Create backup of file before removal"""
        backup_dir = self.project_root / "backup" / "redundant_files"
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        backup_path = backup_dir / file_path.name
        if not self.dry_run:
            shutil.copy2(file_path, backup_path)
            print(f"📦 Backed up: {file_path.name}")
        
        return backup_path
    
    def remove_file(self, file_path: Path) -> bool:
        """Remove a file with backup"""
        try:
            if not self.dry_run:
                # Create backup first
                self.backup_file(file_path)
                
                # Remove the file
                file_path.unlink()
                print(f"🗑️ Removed: {file_path.name}")
                self.removed_files.append(file_path)
            else:
                print(f"🔍 Would remove: {file_path.name}")
                self.redundant_files.append(file_path)
            
            return True
        except Exception as e:
            print(f"❌ Failed to remove {file_path.name}: {e}")
            return False
    
    def remove_directory(self, dir_path: Path) -> bool:
        """Remove a directory with backup"""
        try:
            if not self.dry_run:
                # Create backup first
                backup_dir = self.project_root / "backup" / "redundant_dirs"
                backup_dir.mkdir(parents=True, exist_ok=True)
                
                backup_path = backup_dir / dir_path.name
                shutil.copytree(dir_path, backup_path, dirs_exist_ok=True)
                print(f"📦 Backed up directory: {dir_path.name}")
                
                # Remove the directory
                shutil.rmtree(dir_path)
                print(f"🗑️ Removed directory: {dir_path.name}")
                self.removed_files.append(dir_path)
            else:
                print(f"🔍 Would remove directory: {dir_path.name}")
                self.redundant_files.append(dir_path)
            
            return True
        except Exception as e:
            print(f"❌ Failed to remove directory {dir_path.name}: {e}")
            return False
    
    def cleanup_redundant_files(self) -> Dict:
        """Clean up redundant files"""
        print("🧹 Starting Enterprise-Grade Cleanup")
        print("=" * 50)
        
        # Identify redundant files
        redundant_test_files = self.identify_redundant_test_files()
        debug_files = self.identify_debug_files()
        cache_dirs = self.identify_cache_directories()
        
        total_files = len(redundant_test_files) + len(debug_files) + len(cache_dirs)
        
        if total_files == 0:
            print("✅ No redundant files found!")
            return {"removed": 0, "kept": 0, "errors": 0}
        
        print(f"\n📊 Found {total_files} redundant files/directories:")
        print(f"  - Redundant test files: {len(redundant_test_files)}")
        print(f"  - Debug files: {len(debug_files)}")
        print(f"  - Cache directories: {len(cache_dirs)}")
        
        if self.dry_run:
            print("\n🔍 DRY RUN MODE - No files will be removed")
        else:
            print("\n⚠️ LIVE MODE - Files will be removed and backed up")
        
        # Remove redundant test files
        print(f"\n🗑️ Processing redundant test files...")
        for file_path in redundant_test_files:
            self.remove_file(file_path)
        
        # Remove debug files
        print(f"\n🗑️ Processing debug files...")
        for file_path in debug_files:
            self.remove_file(file_path)
        
        # Remove cache directories
        print(f"\n🗑️ Processing cache directories...")
        for dir_path in cache_dirs:
            self.remove_directory(dir_path)
        
        # Summary
        removed_count = len(self.removed_files) if not self.dry_run else len(self.redundant_files)
        
        print(f"\n📊 Cleanup Summary:")
        print(f"  - Files processed: {total_files}")
        print(f"  - Files {'would be removed' if self.dry_run else 'removed'}: {removed_count}")
        
        return {
            "total_found": total_files,
            "removed": removed_count,
            "dry_run": self.dry_run
        }
    
    def verify_test_structure(self) -> Dict:
        """Verify that proper test structure is in place"""
        print("\n🔍 Verifying Test Structure...")
        print("=" * 50)
        
        test_structure = {
            "tests/unit/": ["test_analytics.py", "test_core_functionality.py"],
            "tests/integration/": ["test_system_integration.py"],
            "tests/e2e/": ["test_complete_workflow.py"],
            "tests/": ["run_tests.py"]
        }
        
        missing_files = []
        existing_files = []
        
        for directory, expected_files in test_structure.items():
            dir_path = self.project_root / directory
            if dir_path.exists():
                for expected_file in expected_files:
                    file_path = dir_path / expected_file
                    if file_path.exists():
                        existing_files.append(f"{directory}{expected_file}")
                        print(f"✅ Found: {directory}{expected_file}")
                    else:
                        missing_files.append(f"{directory}{expected_file}")
                        print(f"❌ Missing: {directory}{expected_file}")
            else:
                print(f"❌ Missing directory: {directory}")
                for expected_file in expected_files:
                    missing_files.append(f"{directory}{expected_file}")
        
        return {
            "existing": existing_files,
            "missing": missing_files,
            "structure_valid": len(missing_files) == 0
        }
    
    def generate_cleanup_report(self, cleanup_results: Dict, test_structure: Dict) -> Dict:
        """Generate comprehensive cleanup report"""
        report = {
            "timestamp": __import__('datetime').datetime.now().isoformat(),
            "cleanup_results": cleanup_results,
            "test_structure": test_structure,
            "recommendations": []
        }
        
        # Generate recommendations
        if cleanup_results["total_found"] > 0:
            report["recommendations"].append(
                f"Removed {cleanup_results['removed']} redundant files to improve project organization"
            )
        
        if not test_structure["structure_valid"]:
            report["recommendations"].append(
                "Test structure needs improvement - some expected test files are missing"
            )
        else:
            report["recommendations"].append(
                "Test structure is properly organized"
            )
        
        if cleanup_results["dry_run"]:
            report["recommendations"].append(
                "Run with --live flag to actually remove files"
            )
        
        return report
    
    def print_report(self, report: Dict):
        """Print cleanup report"""
        print("\n" + "=" * 60)
        print("📊 CLEANUP REPORT")
        print("=" * 60)
        
        cleanup_results = report["cleanup_results"]
        test_structure = report["test_structure"]
        
        print(f"Cleanup Results:")
        print(f"  - Total files found: {cleanup_results['total_found']}")
        print(f"  - Files {'would be removed' if cleanup_results['dry_run'] else 'removed'}: {cleanup_results['removed']}")
        
        print(f"\nTest Structure:")
        print(f"  - Existing test files: {len(test_structure['existing'])}")
        print(f"  - Missing test files: {len(test_structure['missing'])}")
        print(f"  - Structure valid: {'✅ Yes' if test_structure['structure_valid'] else '❌ No'}")
        
        print(f"\nRecommendations:")
        for rec in report["recommendations"]:
            print(f"  - {rec}")
        
        if test_structure["structure_valid"] and cleanup_results["removed"] > 0:
            print("\n🎉 Project cleanup successful! The project is now enterprise-grade.")
        else:
            print("\n⚠️ Some issues remain. Please review the recommendations above.")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="FRED ML Project Cleanup")
    parser.add_argument("--live", action="store_true", help="Actually remove files (default is dry run)")
    parser.add_argument("--verify-only", action="store_true", help="Only verify test structure")
    
    args = parser.parse_args()
    
    cleaner = ProjectCleaner(dry_run=not args.live)
    
    if args.verify_only:
        # Only verify test structure
        test_structure = cleaner.verify_test_structure()
        report = cleaner.generate_cleanup_report({"total_found": 0, "removed": 0, "dry_run": True}, test_structure)
        cleaner.print_report(report)
    else:
        # Full cleanup
        cleanup_results = cleaner.cleanup_redundant_files()
        test_structure = cleaner.verify_test_structure()
        report = cleaner.generate_cleanup_report(cleanup_results, test_structure)
        cleaner.print_report(report)
    
    # Exit with appropriate code
    test_structure = cleaner.verify_test_structure()
    if not test_structure["structure_valid"]:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main() 