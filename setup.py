#!/usr/bin/env python3
"""
Setup script for Herbal Medicine Chatbot
Initializes the project environment and checks dependencies
"""

import subprocess
import sys
import os
from pathlib import Path
import json

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required. Current version:", sys.version)
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def check_herbs_data():
    """Check if herbs.json exists"""
    if not Path("herbs.json").exists():
        print("❌ herbs.json not found!")
        print("Please make sure your herbs dataset is in the current directory.")
        return False
    
    # Check if file is valid JSON
    try:
        with open("herbs.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"✅ herbs.json found with {len(data)} herbs")
        return True
    except json.JSONDecodeError as e:
        print(f"❌ herbs.json is not valid JSON: {e}")
        return False

def install_dependencies():
    """Install Python dependencies"""
    print("📦 Installing Python dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True)
        print("✅ Python dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install Python dependencies: {e}")
        return False

def check_node():
    """Check if Node.js is installed"""
    try:
        result = subprocess.run(["node", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Node.js version: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    
    print("❌ Node.js not found!")
    print("Please install Node.js 16+ from https://nodejs.org/")
    return False

def install_frontend_dependencies():
    """Install frontend dependencies"""
    frontend_path = Path("frontend")
    if not frontend_path.exists():
        print("❌ Frontend directory not found!")
        return False
    
    print("📦 Installing frontend dependencies...")
    try:
        subprocess.run(["npm", "install"], cwd=frontend_path, check=True, capture_output=True)
        print("✅ Frontend dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install frontend dependencies: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    directories = ["data", "herbal_model", "evaluation_results"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    print("✅ Created necessary directories")

def check_gpu():
    """Check if GPU is available"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU available: {gpu_name} (Count: {gpu_count})")
            return True
        else:
            print("ℹ️ No GPU detected, will use CPU training")
            return False
    except ImportError:
        print("ℹ️ PyTorch not installed yet, cannot check GPU")
        return False

def run_quick_test():
    """Run a quick test of the data preparation"""
    print("🧪 Running quick data preparation test...")
    try:
        result = subprocess.run([
            sys.executable, "data_prep.py", 
            "--input", "herbs.json",
            "--output-dir", "data",
            "--qa-per-herb", "2",
            "--subset", "5"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Data preparation test passed")
            return True
        else:
            print(f"❌ Data preparation test failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Data preparation test error: {e}")
        return False

def main():
    """Main setup function"""
    print("🌿 Herbal Medicine Chatbot Setup")
    print("=" * 40)
    
    checks = [
        ("Python Version", check_python_version),
        ("Herbs Data", check_herbs_data),
        ("Node.js", check_node),
        ("GPU Check", check_gpu),
    ]
    
    all_passed = True
    for check_name, check_func in checks:
        print(f"\n🔍 Checking {check_name}...")
        if not check_func():
            all_passed = False
    
    if not all_passed:
        print("\n❌ Some checks failed. Please fix the issues above before continuing.")
        return False
    
    print("\n📦 Installing dependencies...")
    if not install_dependencies():
        return False
    
    if not install_frontend_dependencies():
        return False
    
    print("\n📁 Creating directories...")
    create_directories()
    
    print("\n🧪 Running quick test...")
    if not run_quick_test():
        print("⚠️ Quick test failed, but setup is complete")
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Train the model: python train.py --use-lora --device cpu --subset 100")
    print("2. Evaluate the model: python evaluate.py --model-path herbal_model")
    print("3. Start the API: python api.py")
    print("4. Start the frontend: cd frontend && npm start")
    print("\nFor detailed instructions, see README.md")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)