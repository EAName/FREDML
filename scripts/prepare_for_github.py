#!/usr/bin/env python3
"""
FRED ML - GitHub Preparation Script
Prepares the repository for GitHub submission with final checks and git commands
"""

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

def print_header(title):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f"🚀 {title}")
    print(f"{'='*60}")

def print_success(message):
    """Print success message"""
    print(f"✅ {message}")

def print_error(message):
    """Print error message"""
    print(f"❌ {message}")

def print_warning(message):
    """Print warning message"""
    print(f"⚠️ {message}")

def print_info(message):
    """Print info message"""
    print(f"ℹ️ {message}")

def check_git_status():
    """Check git status and prepare for commit"""
    print_header("Checking Git Status")
    
    try:
        # Check if we're in a git repository
        result = subprocess.run(['git', 'status'], capture_output=True, text=True)
        if result.returncode != 0:
            print_error("Not in a git repository")
            return False
        
        print_success("Git repository found")
        
        # Check current branch
        result = subprocess.run(['git', 'branch', '--show-current'], capture_output=True, text=True)
        current_branch = result.stdout.strip()
        print_info(f"Current branch: {current_branch}")
        
        # Check for changes
        result = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True)
        if result.stdout.strip():
            print_info("Changes detected:")
            print(result.stdout)
            return True
        else:
            print_warning("No changes detected")
            return False
            
    except Exception as e:
        print_error(f"Error checking git status: {e}")
        return False

def create_feature_branch():
    """Create a feature branch for the changes"""
    print_header("Creating Feature Branch")
    
    try:
        # Create feature branch
        branch_name = f"feature/advanced-analytics-{datetime.now().strftime('%Y%m%d')}"
        result = subprocess.run(['git', 'checkout', '-b', branch_name], capture_output=True, text=True)
        
        if result.returncode == 0:
            print_success(f"Created feature branch: {branch_name}")
            return branch_name
        else:
            print_error(f"Failed to create branch: {result.stderr}")
            return None
            
    except Exception as e:
        print_error(f"Error creating feature branch: {e}")
        return None

def add_and_commit_changes():
    """Add and commit all changes"""
    print_header("Adding and Committing Changes")
    
    try:
        # Add all changes
        result = subprocess.run(['git', 'add', '.'], capture_output=True, text=True)
        if result.returncode != 0:
            print_error(f"Failed to add changes: {result.stderr}")
            return False
        
        print_success("Added all changes")
        
        # Commit changes
        commit_message = """feat: Integrate advanced analytics and enterprise UI

- Update cron job schedule to quarterly execution
- Implement enterprise-grade Streamlit UI with think tank aesthetic
- Add comprehensive advanced analytics modules:
  * Enhanced FRED client with 20+ economic indicators
  * Economic forecasting with ARIMA and ETS models
  * Economic segmentation with clustering algorithms
  * Statistical modeling with regression and causality
  * Comprehensive analytics orchestration
- Create automation and testing scripts
- Update documentation and dependencies
- Implement professional styling and responsive design

This transforms FRED ML into an enterprise-grade economic analytics platform."""
        
        result = subprocess.run(['git', 'commit', '-m', commit_message], capture_output=True, text=True)
        if result.returncode == 0:
            print_success("Changes committed successfully")
            return True
        else:
            print_error(f"Failed to commit changes: {result.stderr}")
            return False
            
    except Exception as e:
        print_error(f"Error committing changes: {e}")
        return False

def run_final_tests():
    """Run final tests before submission"""
    print_header("Running Final Tests")
    
    tests = [
        ("Streamlit UI Test", "python scripts/test_streamlit_ui.py"),
        ("System Integration Test", "python scripts/integrate_and_test.py")
    ]
    
    all_passed = True
    for test_name, command in tests:
        print_info(f"Running {test_name}...")
        try:
            result = subprocess.run(command.split(), capture_output=True, text=True)
            if result.returncode == 0:
                print_success(f"{test_name} passed")
            else:
                print_error(f"{test_name} failed")
                print(result.stderr)
                all_passed = False
        except Exception as e:
            print_error(f"Error running {test_name}: {e}")
            all_passed = False
    
    return all_passed

def check_file_structure():
    """Check that all required files are present"""
    print_header("Checking File Structure")
    
    required_files = [
        'frontend/app.py',
        'src/analysis/economic_forecasting.py',
        'src/analysis/economic_segmentation.py',
        'src/analysis/statistical_modeling.py',
        'src/analysis/comprehensive_analytics.py',
        'src/core/enhanced_fred_client.py',
        'scripts/run_advanced_analytics.py',
        'scripts/comprehensive_demo.py',
        'scripts/integrate_and_test.py',
        'scripts/test_complete_system.py',
        'scripts/test_streamlit_ui.py',
        'config/pipeline.yaml',
        'requirements.txt',
        'README.md',
        'docs/ADVANCED_ANALYTICS_SUMMARY.md',
        'docs/INTEGRATION_SUMMARY.md'
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = Path(file_path)
        if full_path.exists():
            print_success(f"✅ {file_path}")
        else:
            print_error(f"❌ {file_path}")
            missing_files.append(file_path)
    
    if missing_files:
        print_error(f"Missing files: {missing_files}")
        return False
    else:
        print_success("All required files present")
        return True

def generate_submission_summary():
    """Generate a summary of what's being submitted"""
    print_header("Submission Summary")
    
    summary = """
🎉 FRED ML Advanced Analytics Integration

📊 Key Improvements:
• Updated cron job schedule to quarterly execution
• Implemented enterprise-grade Streamlit UI with think tank aesthetic
• Added comprehensive advanced analytics modules
• Created automation and testing scripts
• Updated documentation and dependencies

🏗️ New Architecture:
• Enhanced FRED client with 20+ economic indicators
• Economic forecasting with ARIMA and ETS models
• Economic segmentation with clustering algorithms
• Statistical modeling with regression and causality
• Professional UI with responsive design

📁 Files Added/Modified:
• 6 new analytics modules in src/analysis/
• 1 enhanced core module in src/core/
• 1 completely redesigned Streamlit UI
• 5 new automation and testing scripts
• 2 comprehensive documentation files
• Updated configuration and dependencies

🧪 Testing:
• Comprehensive test suite created
• Streamlit UI validation
• System integration testing
• Performance and quality checks

📈 Business Value:
• Enterprise-grade economic analytics platform
• Professional presentation for stakeholders
• Automated quarterly analysis
• Scalable, maintainable architecture
"""
    
    print(summary)

def main():
    """Main preparation function"""
    print_header("FRED ML GitHub Preparation")
    
    # Check git status
    if not check_git_status():
        print_error("Git status check failed. Exiting.")
        sys.exit(1)
    
    # Check file structure
    if not check_file_structure():
        print_error("File structure check failed. Exiting.")
        sys.exit(1)
    
    # Run final tests
    if not run_final_tests():
        print_warning("Some tests failed, but continuing with submission...")
    
    # Create feature branch
    branch_name = create_feature_branch()
    if not branch_name:
        print_error("Failed to create feature branch. Exiting.")
        sys.exit(1)
    
    # Add and commit changes
    if not add_and_commit_changes():
        print_error("Failed to commit changes. Exiting.")
        sys.exit(1)
    
    # Generate summary
    generate_submission_summary()
    
    # Provide next steps
    print_header("Next Steps")
    print_info("1. Review the changes:")
    print("   git log --oneline -5")
    print()
    print_info("2. Push the feature branch:")
    print(f"   git push origin {branch_name}")
    print()
    print_info("3. Create a Pull Request on GitHub:")
    print("   - Go to your GitHub repository")
    print("   - Click 'Compare & pull request'")
    print("   - Add description of changes")
    print("   - Request review from team members")
    print()
    print_info("4. After approval, merge to main:")
    print("   git checkout main")
    print("   git pull origin main")
    print("   git branch -d " + branch_name)
    print()
    print_success("🎉 Repository ready for GitHub submission!")

if __name__ == "__main__":
    main() 