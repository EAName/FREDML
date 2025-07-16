#!/usr/bin/env python3
"""
Virtual Environment Setup Script for FRED ML
Creates and configures a virtual environment for development
"""

import os
import sys
import subprocess
import venv
from pathlib import Path


def create_venv(venv_path: str = ".venv") -> bool:
    """Create a virtual environment"""
    try:
        print(f"Creating virtual environment at {venv_path}...")
        venv.create(venv_path, with_pip=True)
        print("✅ Virtual environment created successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to create virtual environment: {e}")
        return False


def install_requirements(venv_path: str = ".venv") -> bool:
    """Install requirements in the virtual environment"""
    try:
        # Determine the pip path
        if os.name == 'nt':  # Windows
            pip_path = os.path.join(venv_path, "Scripts", "pip")
        else:  # Unix/Linux/macOS
            pip_path = os.path.join(venv_path, "bin", "pip")
        
        print("Installing requirements...")
        subprocess.run([pip_path, "install", "-r", "requirements.txt"], check=True)
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error installing requirements: {e}")
        return False


def activate_venv_instructions(venv_path: str = ".venv"):
    """Print activation instructions"""
    print("\n📋 Virtual Environment Setup Complete!")
    print("=" * 50)
    
    if os.name == 'nt':  # Windows
        activate_script = os.path.join(venv_path, "Scripts", "activate")
        print(f"To activate the virtual environment, run:")
        print(f"  {activate_script}")
    else:  # Unix/Linux/macOS
        activate_script = os.path.join(venv_path, "bin", "activate")
        print(f"To activate the virtual environment, run:")
        print(f"  source {activate_script}")
    
    print("\nOr use the provided Makefile target:")
    print("  make venv-activate")
    
    print("\nTo deactivate, simply run:")
    print("  deactivate")


def main():
    """Main setup function"""
    print("🏗️ FRED ML - Virtual Environment Setup")
    print("=" * 40)
    
    venv_path = ".venv"
    
    # Check if virtual environment already exists
    if os.path.exists(venv_path):
        print(f"⚠️ Virtual environment already exists at {venv_path}")
        response = input("Do you want to recreate it? (y/N): ").lower().strip()
        if response == 'y':
            import shutil
            shutil.rmtree(venv_path)
            print("Removed existing virtual environment")
        else:
            print("Using existing virtual environment")
            activate_venv_instructions(venv_path)
            return
    
    # Create virtual environment
    if not create_venv(venv_path):
        sys.exit(1)
    
    # Install requirements
    if not install_requirements(venv_path):
        print("⚠️ Failed to install requirements, but virtual environment was created")
        print("You can manually install requirements after activation")
    
    # Print activation instructions
    activate_venv_instructions(venv_path)


if __name__ == "__main__":
    main() 