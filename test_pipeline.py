#!/usr/bin/env python3
"""
End-to-end pipeline test for Herbal Medicine Chatbot
Tests the complete workflow from data preparation to API serving
"""

import json
import subprocess
import sys
import time
import requests
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PipelineTester:
    def __init__(self):
        self.test_results = {}
        self.api_url = "http://localhost:8000"
        
    def run_command(self, command, description):
        """Run a command and return success status"""
        logger.info(f"Running: {description}")
        logger.info(f"Command: {command}")
        
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            success = result.returncode == 0
            
            if success:
                logger.info(f"✅ {description} - SUCCESS")
            else:
                logger.error(f"❌ {description} - FAILED")
                logger.error(f"Error: {result.stderr}")
            
            return success, result.stdout, result.stderr
            
        except Exception as e:
            logger.error(f"❌ {description} - EXCEPTION: {e}")
            return False, "", str(e)
    
    def test_data_preparation(self):
        """Test data preparation pipeline"""
        logger.info("🧪 Testing data preparation...")
        
        # Check if herbs.json exists
        if not Path("herbs.json").exists():
            logger.error("❌ herbs.json not found!")
            return False
        
        # Run data preparation
        success, stdout, stderr = self.run_command(
            "python data_prep.py --input herbs.json --output-dir data --qa-per-herb 5 --subset 10",
            "Data preparation"
        )
        
        if not success:
            return False
        
        # Check if output files exist
        required_files = ["data/train.jsonl", "data/test.jsonl"]
        for file_path in required_files:
            if not Path(file_path).exists():
                logger.error(f"❌ Required file not created: {file_path}")
                return False
        
        # Check file contents
        with open("data/train.jsonl", "r") as f:
            train_lines = len(f.readlines())
        
        with open("data/test.jsonl", "r") as f:
            test_lines = len(f.readlines())
        
        logger.info(f"✅ Generated {train_lines} training examples and {test_lines} test examples")
        
        self.test_results["data_preparation"] = {
            "success": True,
            "train_samples": train_lines,
            "test_samples": test_lines
        }
        
        return True
    
    def test_model_training(self):
        """Test model training pipeline"""
        logger.info("🧪 Testing model training...")
        
        # Run training with minimal configuration
        success, stdout, stderr = self.run_command(
            "python train.py --model distilgpt2 --train-data data/train.jsonl --eval-data data/test.jsonl --output-dir herbal_model --epochs 1 --batch-size 1 --use-lora --device cpu --subset 5",
            "Model training"
        )
        
        if not success:
            return False
        
        # Check if model files exist
        model_path = Path("herbal_model")
        if not model_path.exists():
            logger.error("❌ Model directory not created!")
            return False
        
        # Check for key model files
        required_files = ["config.json", "training_args.bin"]
        for file_path in required_files:
            if not (model_path / file_path).exists():
                logger.warning(f"⚠️ Model file not found: {file_path}")
        
        logger.info("✅ Model training completed")
        
        self.test_results["model_training"] = {
            "success": True,
            "model_path": str(model_path)
        }
        
        return True
    
    def test_model_evaluation(self):
        """Test model evaluation pipeline"""
        logger.info("🧪 Testing model evaluation...")
        
        # Run evaluation
        success, stdout, stderr = self.run_command(
            "python evaluate.py --model-path herbal_model --base-model distilgpt2 --test-data data/test.jsonl --output-dir evaluation_results --num-samples 5",
            "Model evaluation"
        )
        
        if not success:
            return False
        
        # Check if evaluation results exist
        eval_path = Path("evaluation_results")
        if not eval_path.exists():
            logger.error("❌ Evaluation results directory not created!")
            return False
        
        # Check for key evaluation files
        required_files = ["evaluation_report.json", "summary_report.md"]
        for file_path in required_files:
            if not (eval_path / file_path).exists():
                logger.warning(f"⚠️ Evaluation file not found: {file_path}")
        
        logger.info("✅ Model evaluation completed")
        
        self.test_results["model_evaluation"] = {
            "success": True,
            "eval_path": str(eval_path)
        }
        
        return True
    
    def test_api_server(self):
        """Test API server functionality"""
        logger.info("🧪 Testing API server...")
        
        # Start API server in background
        logger.info("Starting API server...")
        api_process = subprocess.Popen(
            ["python", "api.py", "--host", "127.0.0.1", "--port", "8000"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for server to start
        time.sleep(5)
        
        try:
            # Test health endpoint
            response = requests.get(f"{self.api_url}/health", timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                logger.info(f"✅ API health check passed: {health_data}")
            else:
                logger.error(f"❌ API health check failed: {response.status_code}")
                return False
            
            # Test chat endpoint
            chat_data = {
                "message": "What is Sokhrus?",
                "max_length": 100,
                "temperature": 0.7
            }
            
            response = requests.post(f"{self.api_url}/chat", json=chat_data, timeout=30)
            if response.status_code == 200:
                chat_response = response.json()
                logger.info(f"✅ Chat endpoint working: {chat_response['response'][:100]}...")
            else:
                logger.error(f"❌ Chat endpoint failed: {response.status_code}")
                return False
            
            # Test model info endpoint
            response = requests.get(f"{self.api_url}/model-info", timeout=10)
            if response.status_code == 200:
                model_info = response.json()
                logger.info(f"✅ Model info endpoint working: {model_info}")
            else:
                logger.error(f"❌ Model info endpoint failed: {response.status_code}")
                return False
            
            logger.info("✅ API server tests passed")
            
            self.test_results["api_server"] = {
                "success": True,
                "health_status": health_data,
                "chat_response_length": len(chat_response['response'])
            }
            
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ API server test failed: {e}")
            return False
        
        finally:
            # Stop API server
            api_process.terminate()
            api_process.wait()
            logger.info("API server stopped")
    
    def test_frontend_build(self):
        """Test frontend build process"""
        logger.info("🧪 Testing frontend build...")
        
        frontend_path = Path("frontend")
        if not frontend_path.exists():
            logger.error("❌ Frontend directory not found!")
            return False
        
        # Install dependencies
        success, stdout, stderr = self.run_command(
            "cd frontend && npm install",
            "Frontend dependency installation"
        )
        
        if not success:
            logger.error("❌ Frontend dependency installation failed!")
            return False
        
        # Build frontend
        success, stdout, stderr = self.run_command(
            "cd frontend && npm run build",
            "Frontend build"
        )
        
        if not success:
            logger.error("❌ Frontend build failed!")
            return False
        
        # Check if build directory exists
        build_path = frontend_path / "build"
        if not build_path.exists():
            logger.error("❌ Frontend build directory not created!")
            return False
        
        logger.info("✅ Frontend build completed")
        
        self.test_results["frontend_build"] = {
            "success": True,
            "build_path": str(build_path)
        }
        
        return True
    
    def run_full_pipeline_test(self):
        """Run complete pipeline test"""
        logger.info("🚀 Starting full pipeline test...")
        
        tests = [
            ("Data Preparation", self.test_data_preparation),
            ("Model Training", self.test_model_training),
            ("Model Evaluation", self.test_model_evaluation),
            ("API Server", self.test_api_server),
            ("Frontend Build", self.test_frontend_build),
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            logger.info(f"\n{'='*50}")
            logger.info(f"Running: {test_name}")
            logger.info(f"{'='*50}")
            
            try:
                if test_func():
                    passed += 1
                    logger.info(f"✅ {test_name} - PASSED")
                else:
                    logger.error(f"❌ {test_name} - FAILED")
            except Exception as e:
                logger.error(f"❌ {test_name} - EXCEPTION: {e}")
        
        logger.info(f"\n{'='*50}")
        logger.info(f"PIPELINE TEST RESULTS: {passed}/{total} tests passed")
        logger.info(f"{'='*50}")
        
        if passed == total:
            logger.info("🎉 All tests passed! Pipeline is working correctly.")
        else:
            logger.error(f"❌ {total - passed} tests failed. Please check the errors above.")
        
        # Save test results
        with open("pipeline_test_results.json", "w") as f:
            json.dump(self.test_results, f, indent=2)
        
        return passed == total

def main():
    """Main test function"""
    tester = PipelineTester()
    
    if len(sys.argv) > 1:
        test_type = sys.argv[1].lower()
        
        if test_type == "data":
            success = tester.test_data_preparation()
        elif test_type == "train":
            success = tester.test_model_training()
        elif test_type == "eval":
            success = tester.test_model_evaluation()
        elif test_type == "api":
            success = tester.test_api_server()
        elif test_type == "frontend":
            success = tester.test_frontend_build()
        else:
            logger.error(f"Unknown test type: {test_type}")
            logger.info("Available test types: data, train, eval, api, frontend, full")
            sys.exit(1)
        
        sys.exit(0 if success else 1)
    else:
        # Run full pipeline test
        success = tester.run_full_pipeline_test()
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
