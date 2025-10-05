#!/usr/bin/env python3
"""
Test script to verify the herbal medicine chatbot setup
"""

import os
import sys
import json
import subprocess
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists"""
    if Path(filepath).exists():
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description} missing: {filepath}")
        return False

def check_directory_exists(dirpath, description):
    """Check if a directory exists"""
    if Path(dirpath).exists() and Path(dirpath).is_dir():
        print(f"✅ {description}: {dirpath}")
        return True
    else:
        print(f"❌ {description} missing: {dirpath}")
        return False

def check_python_imports():
    """Check if required Python packages can be imported"""
    required_packages = [
        'torch',
        'transformers',
        'datasets',
        'peft',
        'fastapi',
        'uvicorn',
        'pandas',
        'numpy'
    ]
    
    print("\n🐍 Checking Python packages...")
    all_good = True
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} not installed")
            all_good = False
    
    return all_good

def check_dataset():
    """Check if dataset files exist and are valid"""
    print("\n📊 Checking dataset...")
    
    # Check original dataset
    if not check_file_exists("herbs.json", "Original herbs dataset"):
        return False
    
    # Check if processed datasets exist
    data_dir = Path("data")
    if data_dir.exists():
        files_to_check = [
            "data/qa_pairs.jsonl",
            "data/train_dataset.jsonl", 
            "data/val_dataset.jsonl"
        ]
        
        for file in files_to_check:
            check_file_exists(file, f"Processed dataset")
    else:
        print("⚠️ Data directory not found - run 'python src/data_prep.py' first")
    
    return True

def check_models():
    """Check if trained models exist"""
    print("\n🤖 Checking models...")
    
    models_dir = Path("models")
    if models_dir.exists():
        model_files = list(models_dir.glob("*"))
        if model_files:
            print(f"✅ Found {len(model_files)} model files/directories")
            for model in model_files:
                print(f"   📁 {model.name}")
        else:
            print("⚠️ No trained models found - run 'python src/train.py' to train a model")
    else:
        print("⚠️ Models directory not found - will be created during training")
    
    return True

def check_frontend():
    """Check if frontend is set up"""
    print("\n⚛️ Checking frontend...")
    
    frontend_dir = Path("frontend")
    if not check_directory_exists("frontend", "Frontend directory"):
        return False
    
    # Check package.json
    if not check_file_exists("frontend/package.json", "Frontend package.json"):
        return False
    
    # Check if node_modules exists
    node_modules = frontend_dir / "node_modules"
    if node_modules.exists():
        print("✅ Frontend dependencies installed")
    else:
        print("⚠️ Frontend dependencies not installed - run 'cd frontend && npm install'")
    
    # Check main React files
    react_files = [
        "frontend/src/App.js",
        "frontend/src/App.css",
        "frontend/src/index.js"
    ]
    
    for file in react_files:
        check_file_exists(file, "React component")
    
    return True

def check_scripts():
    """Check if utility scripts exist"""
    print("\n📝 Checking utility scripts...")
    
    scripts = [
        ("src/data_prep.py", "Dataset preparation script"),
        ("src/train.py", "Model training script"),
        ("src/evaluate.py", "Model evaluation script"),
        ("src/api.py", "API server script"),
        ("setup.py", "Setup script"),
        ("run_example.py", "Example test script")
    ]
    
    all_good = True
    for script, description in scripts:
        if not check_file_exists(script, description):
            all_good = False
    
    return all_good

def test_data_preparation():
    """Test if data preparation works"""
    print("\n🧪 Testing data preparation...")
    
    try:
        # Test with a small subset
        result = subprocess.run([
            sys.executable, "src/data_prep.py", "--max_herbs", "5"
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ Data preparation test passed")
            return True
        else:
            print(f"❌ Data preparation test failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Data preparation test timed out")
        return False
    except Exception as e:
        print(f"❌ Data preparation test error: {e}")
        return False

def test_api_import():
    """Test if API can be imported"""
    print("\n🌐 Testing API import...")
    
    try:
        # Change to src directory and try importing
        sys.path.insert(0, 'src')
        import api
        print("✅ API import test passed")
        return True
    except Exception as e:
        print(f"❌ API import test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Herbal Medicine Chatbot - Setup Verification")
    print("=" * 50)
    
    tests = [
        ("File Structure", lambda: (
            check_file_exists("requirements.txt", "Requirements file") and
            check_file_exists("README.md", "README file") and
            check_directory_exists("src", "Source directory") and
            check_scripts()
        )),
        ("Python Packages", check_python_imports),
        ("Dataset", check_dataset),
        ("Models", check_models),
        ("Frontend", check_frontend),
        ("Data Preparation", test_data_preparation),
        ("API Import", test_api_import)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test failed with error: {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 Test Summary:")
    print(f"{'='*50}")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name:<20} {status}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your setup is ready.")
        print("\n🚀 Next steps:")
        print("   1. Train a model: python src/train.py --max_samples 500")
        print("   2. Start API: python src/api.py")
        print("   3. Start frontend: cd frontend && npm start")
        print("   4. Test chatbot: python run_example.py")
    else:
        print(f"\n⚠️ {total - passed} tests failed. Please fix the issues above.")
        print("\n💡 Common fixes:")
        print("   - Install dependencies: pip install -r requirements.txt")
        print("   - Prepare dataset: python src/data_prep.py")
        print("   - Install frontend: cd frontend && npm install")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)