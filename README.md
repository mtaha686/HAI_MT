# 🌿 Herbal Medicine Chatbot

A complete end-to-end chatbot system for herbal medicine information using fine-tuned language models. Built with Python, Hugging Face Transformers, FastAPI, and React.

## 🚀 One-Click Setup

**Automated Setup (Recommended):**
```bash
python setup.py
```

This will automatically:
- Create virtual environment
- Install all dependencies
- Prepare the dataset
- Optionally train the model
- Setup the React frontend
- Create startup scripts

## 📁 Project Structure

```
herbal-chatbot/
├── herbs.json                    # Your original dataset (4000+ herbs)
├── data/                         # Generated datasets
│   ├── qa_pairs.jsonl           # Generated Q&A pairs
│   ├── train_dataset.jsonl      # Training data
│   └── val_dataset.jsonl        # Validation data
├── src/                         # Python source code
│   ├── data_prep.py            # Dataset preparation
│   ├── train.py                # Model fine-tuning
│   ├── evaluate.py             # Model evaluation
│   └── api.py                  # FastAPI backend
├── models/                     # Saved models
├── frontend/                   # React chatbot UI
│   ├── src/
│   │   ├── App.js             # Main React component
│   │   └── App.css            # Styling
│   └── package.json           # Frontend dependencies
├── requirements.txt            # Python dependencies
├── setup.py                   # Automated setup script
├── run_example.py             # Test script
├── start_chatbot.bat          # Windows startup script
├── start_chatbot.sh           # Unix/Linux/Mac startup script
└── README.md                  # This file
```

## 🔧 Manual Setup (Alternative)

### Prerequisites
- Python 3.8+ 
- Node.js 14+ (for React frontend)
- 8GB+ RAM (16GB recommended)
- 10GB+ free disk space

### Step 1: Python Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Unix/Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Prepare Dataset
```bash
# Generate Q&A pairs from your herbs dataset
python src/data_prep.py

# For subset (faster testing):
python src/data_prep.py --max_herbs 100
```

### Step 3: Train Model

**CPU Training (Recommended for testing):**
```bash
python src/train.py --model_name distilgpt2 --max_samples 500 --epochs 2 --batch_size 2
```

**GPU Training (Better quality):**
```bash
python src/train.py --model_name microsoft/DialoGPT-medium --max_samples 2000 --epochs 3 --batch_size 8
```

**Full Dataset Training:**
```bash
python src/train.py --model_name distilgpt2 --epochs 5 --batch_size 4
```

### Step 4: Start API Server
```bash
python src/api.py

# Or with custom model:
python src/api.py --model_path models/your_trained_model
```

### Step 5: Setup Frontend
```bash
cd frontend
npm install
npm start
```

## 🎯 Quick Start (After Setup)

**Option 1: Use Startup Scripts**
```bash
# Windows:
start_chatbot.bat

# Unix/Linux/Mac:
./start_chatbot.sh
```

**Option 2: Manual Start**
```bash
# Terminal 1 - API Server:
python src/api.py

# Terminal 2 - Frontend:
cd frontend && npm start
```

**Option 3: Test with Example Script**
```bash
python run_example.py
```

## 🌐 Access Points

- **Frontend UI**: http://localhost:3000
- **API Server**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 💻 System Requirements

### Minimum (CPU Only)
- **RAM**: 8GB
- **Storage**: 10GB free space
- **Model**: distilGPT-2 (82M parameters)
- **Training Time**: 10-30 minutes for 500 samples

### Recommended (GPU)
- **RAM**: 16GB
- **GPU**: 4GB+ VRAM
- **Storage**: 20GB free space
- **Model**: DialoGPT-medium (117M parameters)
- **Training Time**: 5-15 minutes for 2000 samples

### Full Scale (Production)
- **RAM**: 32GB
- **GPU**: 8GB+ VRAM
- **Storage**: 50GB free space
- **Model**: GPT-Neo-1.3B or LLaMA-2-7B
- **Training Time**: 1-3 hours for full dataset

## 🤖 Model Options

| Model | Parameters | Memory | Quality | Speed | Best For |
|-------|------------|--------|---------|-------|----------|
| distilGPT-2 | 82M | 2GB | Good | Fast | CPU, Testing |
| GPT-Neo-125M | 125M | 3GB | Better | Medium | CPU/GPU |
| DialoGPT-medium | 117M | 4GB | Better | Medium | GPU |
| GPT-Neo-1.3B | 1.3B | 8GB | Excellent | Slow | GPU Only |

## 📊 Features

### Dataset Processing
- ✅ Automatic Q&A pair generation (10 per herb)
- ✅ Multiple question formats and variations
- ✅ JSONL format for training compatibility
- ✅ Train/validation split (90/10)
- ✅ Support for 4000+ herbs dataset

### Model Training
- ✅ LoRA (Low-Rank Adaptation) for memory efficiency
- ✅ Parameter-efficient fine-tuning
- ✅ CPU and GPU support
- ✅ Automatic model saving and loading
- ✅ Training progress monitoring

### API Backend
- ✅ FastAPI with automatic documentation
- ✅ CORS enabled for frontend integration
- ✅ Health check endpoints
- ✅ Confidence scoring
- ✅ Response time tracking
- ✅ Error handling and fallbacks

### Frontend Interface
- ✅ Modern React chat interface
- ✅ Real-time messaging
- ✅ Sample questions for easy testing
- ✅ Response confidence display
- ✅ Mobile-responsive design
- ✅ Typing indicators and animations

### Evaluation & Monitoring
- ✅ Perplexity calculation
- ✅ Response quality comparison
- ✅ Performance metrics
- ✅ Model comparison tools

## 🧪 Usage Examples

### API Testing
```bash
# Health check
curl http://localhost:8000/health

# Ask a question
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"message": "What are the uses of Sokhrus?"}'

# Get model information
curl http://localhost:8000/model-info
```

### Python Integration
```python
import requests

response = requests.post("http://localhost:8000/chat", 
    json={"message": "How do you prepare Chamomile?"})
print(response.json()["response"])
```

### Sample Questions to Try
- "What are the uses of Sokhrus?"
- "How do you prepare Chamomile tea?"
- "What are the side effects of Astragalus?"
- "Tell me about Rhodiola imbricata"
- "Which parts of Equisetum are used medicinally?"
- "What family does Capparis spinosa belong to?"
- "Is Nepeta erecta safe to use?"
- "Where is Primula macrophylla found?"

## 🔄 Scaling to Full Dataset

### For 4000+ Herbs (Full Production)
```bash
# 1. Prepare full dataset
python src/data_prep.py

# 2. Train with larger model (requires GPU)
python src/train.py --model_name microsoft/DialoGPT-medium --epochs 5 --batch_size 8

# 3. Evaluate performance
python src/evaluate.py --fine_tuned_path models/your_model

# 4. Deploy with trained model
python src/api.py --model_path models/your_trained_model
```

### Memory Optimization Tips
```bash
# Use gradient accumulation for larger effective batch size
python src/train.py --batch_size 2 --gradient_accumulation_steps 4

# Use LoRA with smaller rank for less memory
python src/train.py --lora_r 8 --lora_alpha 16

# Train on subset first, then full dataset
python src/train.py --max_samples 1000 --epochs 3
```

## 🔧 Troubleshooting

### Common Issues

**1. Out of Memory Error**
```bash
# Reduce batch size
python src/train.py --batch_size 1

# Use smaller model
python src/train.py --model_name distilgpt2
```

**2. API Connection Error**
```bash
# Check if API is running
curl http://localhost:8000/health

# Start API server
python src/api.py
```

**3. Frontend Not Loading**
```bash
# Install dependencies
cd frontend && npm install

# Start development server
npm start
```

**4. Model Loading Error**
```bash
# Use base model as fallback
python src/api.py --model_path ""

# Check model path exists
ls models/
```

### Performance Optimization

**For CPU Systems:**
- Use distilGPT-2 model
- Batch size 1-2
- Max samples 500-1000
- Enable gradient accumulation

**For GPU Systems:**
- Use DialoGPT-medium or larger
- Batch size 4-8
- Full dataset training
- Enable mixed precision (fp16)

## 📈 Evaluation & Monitoring

### Evaluate Model Performance
```bash
# Compare base vs fine-tuned model
python src/evaluate.py --base_model distilgpt2 --fine_tuned_path models/your_model

# Test specific questions
python run_example.py
```

### Monitor Training
- Check `models/logs/` for training logs
- Monitor loss curves and perplexity
- Use validation set for early stopping

## 🔄 Updating Dataset

### When You Add New Herbs
```bash
# 1. Update herbs.json with new data
# 2. Regenerate Q&A pairs
python src/data_prep.py

# 3. Retrain model with new data
python src/train.py --model_name your_previous_model --epochs 2

# 4. Evaluate improvements
python src/evaluate.py
```

## ⚠️ Important Notes

### Safety & Disclaimers
- This chatbot provides **educational information only**
- Always consult healthcare professionals before using herbal remedies
- Verify information with authoritative sources
- Not intended for medical diagnosis or treatment

### Data Privacy
- All processing happens locally on your machine
- No data is sent to external services
- Your herbs dataset remains private

### Model Limitations
- Responses based on training data only
- May generate plausible but incorrect information
- Quality depends on training data and model size
- Always verify critical information

## 🤝 Contributing

### Adding New Features
1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Submit pull request

### Improving Model Performance
- Add more diverse Q&A pairs
- Experiment with different models
- Tune hyperparameters
- Add evaluation metrics

## 📚 Additional Resources

### Documentation
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://reactjs.org/docs)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)

### Model Repositories
- [distilGPT-2](https://huggingface.co/distilgpt2)
- [DialoGPT](https://huggingface.co/microsoft/DialoGPT-medium)
- [GPT-Neo](https://huggingface.co/EleutherAI/gpt-neo-125M)

## 🎉 Success Indicators

After successful setup, you should have:
- ✅ API server running on http://localhost:8000
- ✅ React frontend on http://localhost:3000
- ✅ Trained model responding to herb questions
- ✅ Confidence scores and response times displayed
- ✅ Sample questions working correctly

**Ready to explore herbal medicine with AI! 🌿🤖**