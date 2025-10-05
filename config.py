#!/usr/bin/env python3
"""
Configuration file for Herbal Medicine Chatbot
Contains all configurable parameters for training, evaluation, and deployment
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "herbal_model"
EVAL_DIR = BASE_DIR / "evaluation_results"

# Data configuration
DATA_CONFIG = {
    "input_file": "herbs.json",
    "output_dir": str(DATA_DIR),
    "qa_per_herb": 10,
    "train_ratio": 0.8,
    "max_length": 512,
    "subset_size": None,  # Set to integer for testing with subset
}

# Model configuration
MODEL_CONFIG = {
    "base_model": "distilgpt2",  # Options: distilgpt2, gpt2, microsoft/DialoGPT-small
    "use_lora": True,
    "lora_config": {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "target_modules": ["q_lin", "v_lin"],  # For distilgpt2
    },
    "device": "auto",  # auto, cpu, cuda
}

# Training configuration
TRAINING_CONFIG = {
    "num_epochs": 3,
    "batch_size": 4,
    "learning_rate": 5e-5,
    "warmup_steps": 100,
    "logging_steps": 10,
    "eval_steps": 100,
    "save_steps": 500,
    "gradient_accumulation_steps": 4,  # For CPU training
    "fp16": False,  # Set to True for GPU
    "max_length": 512,
    "early_stopping_patience": 3,
}

# Evaluation configuration
EVALUATION_CONFIG = {
    "num_samples": 50,
    "max_length": 200,
    "temperature": 0.7,
    "top_p": 0.9,
    "perplexity_samples": 100,
}

# API configuration
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "max_length": 200,
    "temperature": 0.7,
    "top_p": 0.9,
    "reload": False,  # Set to True for development
}

# Frontend configuration
FRONTEND_CONFIG = {
    "api_url": "http://localhost:8000",
    "port": 3000,
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "use_wandb": False,  # Set to True to use Weights & Biases
    "wandb_project": "herbal-medicine-chatbot",
}

# Memory optimization for different system configurations
MEMORY_CONFIGS = {
    "low_memory": {  # For systems with < 8GB RAM
        "batch_size": 2,
        "gradient_accumulation_steps": 8,
        "max_length": 256,
        "use_lora": True,
        "fp16": False,
    },
    "medium_memory": {  # For systems with 8-16GB RAM
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
        "max_length": 512,
        "use_lora": True,
        "fp16": False,
    },
    "high_memory": {  # For systems with > 16GB RAM or GPU
        "batch_size": 8,
        "gradient_accumulation_steps": 2,
        "max_length": 512,
        "use_lora": False,
        "fp16": True,
    }
}

def get_memory_config(system_type: str = "medium_memory") -> dict:
    """Get memory-optimized configuration based on system type"""
    return MEMORY_CONFIGS.get(system_type, MEMORY_CONFIGS["medium_memory"])

def update_config_from_args(args):
    """Update configuration from command line arguments"""
    if hasattr(args, 'batch_size') and args.batch_size:
        TRAINING_CONFIG["batch_size"] = args.batch_size
    
    if hasattr(args, 'epochs') and args.epochs:
        TRAINING_CONFIG["num_epochs"] = args.epochs
    
    if hasattr(args, 'learning_rate') and args.learning_rate:
        TRAINING_CONFIG["learning_rate"] = args.learning_rate
    
    if hasattr(args, 'use_lora') and args.use_lora:
        MODEL_CONFIG["use_lora"] = args.use_lora
    
    if hasattr(args, 'device') and args.device:
        MODEL_CONFIG["device"] = args.device

# Environment-specific configurations
def get_config_for_environment(env: str = "development"):
    """Get configuration for specific environment"""
    configs = {
        "development": {
            "subset_size": 100,  # Use subset for quick testing
            "num_epochs": 1,
            "eval_steps": 50,
            "use_wandb": False,
            "reload": True,
        },
        "production": {
            "subset_size": None,  # Use full dataset
            "num_epochs": 5,
            "eval_steps": 200,
            "use_wandb": True,
            "reload": False,
        },
        "colab": {
            "batch_size": 8,
            "fp16": True,
            "use_lora": False,
            "device": "cuda",
            "num_epochs": 3,
        }
    }
    
    return configs.get(env, configs["development"])

# Create directories if they don't exist
def create_directories():
    """Create necessary directories"""
    directories = [DATA_DIR, MODEL_DIR, EVAL_DIR]
    for directory in directories:
        directory.mkdir(exist_ok=True)

if __name__ == "__main__":
    # Print current configuration
    print("Current Configuration:")
    print(f"Data Config: {DATA_CONFIG}")
    print(f"Model Config: {MODEL_CONFIG}")
    print(f"Training Config: {TRAINING_CONFIG}")
    print(f"API Config: {API_CONFIG}")
    
    # Create directories
    create_directories()
    print("Directories created successfully!")
