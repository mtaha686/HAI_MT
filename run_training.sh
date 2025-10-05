#!/bin/bash

# Herbal Medicine Chatbot Training Script
# This script provides different training configurations for various environments

echo "🌿 Herbal Medicine Chatbot Training Script"
echo "=========================================="

# Check if herbs.json exists
if [ ! -f "herbs.json" ]; then
    echo "❌ Error: herbs.json not found!"
    echo "Please make sure your herbs dataset is in the current directory."
    exit 1
fi

# Function to run data preparation
prepare_data() {
    echo "📊 Preparing training data..."
    python data_prep.py \
        --input herbs.json \
        --output-dir data \
        --qa-per-herb 10 \
        --train-ratio 0.8 \
        --subset 100  # Use subset for quick testing
    echo "✅ Data preparation complete!"
}

# Function to run training
run_training() {
    local config=$1
    echo "🚀 Starting training with $config configuration..."
    
    case $config in
        "cpu")
            python train.py \
                --model distilgpt2 \
                --train-data data/train.jsonl \
                --eval-data data/test.jsonl \
                --output-dir herbal_model \
                --epochs 3 \
                --batch-size 2 \
                --learning-rate 5e-5 \
                --use-lora \
                --device cpu \
                --subset 50
            ;;
        "gpu")
            python train.py \
                --model distilgpt2 \
                --train-data data/train.jsonl \
                --eval-data data/test.jsonl \
                --output-dir herbal_model \
                --epochs 5 \
                --batch-size 8 \
                --learning-rate 5e-5 \
                --device cuda \
                --subset 200
            ;;
        "colab")
            python train.py \
                --model distilgpt2 \
                --train-data data/train.jsonl \
                --eval-data data/test.jsonl \
                --output-dir herbal_model \
                --epochs 3 \
                --batch-size 4 \
                --learning-rate 5e-5 \
                --use-lora \
                --device cuda
            ;;
        "full")
            python train.py \
                --model distilgpt2 \
                --train-data data/train.jsonl \
                --eval-data data/test.jsonl \
                --output-dir herbal_model \
                --epochs 5 \
                --batch-size 4 \
                --learning-rate 5e-5 \
                --use-lora \
                --device auto
            ;;
        *)
            echo "❌ Unknown configuration: $config"
            echo "Available configurations: cpu, gpu, colab, full"
            exit 1
            ;;
    esac
    
    echo "✅ Training complete!"
}

# Function to run evaluation
run_evaluation() {
    echo "📈 Running evaluation..."
    python evaluate.py \
        --model-path herbal_model \
        --base-model distilgpt2 \
        --test-data data/test.jsonl \
        --output-dir evaluation_results \
        --num-samples 50
    echo "✅ Evaluation complete!"
}

# Function to start API server
start_api() {
    echo "🌐 Starting API server..."
    python api.py --host 0.0.0.0 --port 8000
}

# Function to start frontend
start_frontend() {
    echo "🎨 Starting frontend..."
    cd frontend
    npm install
    npm start
}

# Main script logic
case $1 in
    "prepare")
        prepare_data
        ;;
    "train")
        if [ -z "$2" ]; then
            echo "❌ Please specify training configuration: cpu, gpu, colab, or full"
            echo "Usage: $0 train <config>"
            exit 1
        fi
        prepare_data
        run_training $2
        ;;
    "eval")
        run_evaluation
        ;;
    "api")
        start_api
        ;;
    "frontend")
        start_frontend
        ;;
    "full")
        echo "🚀 Running full pipeline..."
        prepare_data
        run_training "cpu"  # Default to CPU for safety
        run_evaluation
        echo "✅ Full pipeline complete!"
        echo "🎯 Next steps:"
        echo "   1. Start API: $0 api"
        echo "   2. Start Frontend: $0 frontend"
        ;;
    *)
        echo "🌿 Herbal Medicine Chatbot - Usage Guide"
        echo "========================================"
        echo ""
        echo "Available commands:"
        echo "  prepare          - Prepare training data from herbs.json"
        echo "  train <config>   - Train model with specified configuration"
        echo "                     Configs: cpu, gpu, colab, full"
        echo "  eval             - Evaluate trained model"
        echo "  api              - Start API server"
        echo "  frontend         - Start React frontend"
        echo "  full             - Run complete pipeline (prepare + train + eval)"
        echo ""
        echo "Examples:"
        echo "  $0 prepare                    # Prepare data only"
        echo "  $0 train cpu                  # Train on CPU (recommended for local)"
        echo "  $0 train gpu                  # Train on GPU (if available)"
        echo "  $0 train colab                # Train for Google Colab"
        echo "  $0 train full                 # Train with full dataset"
        echo "  $0 eval                        # Evaluate model"
        echo "  $0 api                         # Start API server"
        echo "  $0 frontend                    # Start frontend"
        echo "  $0 full                        # Run everything"
        echo ""
        echo "System Requirements:"
        echo "  - Python 3.8+"
        echo "  - 8GB+ RAM (for CPU training)"
        echo "  - CUDA-compatible GPU (optional, for GPU training)"
        echo "  - Node.js 16+ (for frontend)"
        ;;
esac
