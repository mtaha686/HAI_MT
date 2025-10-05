#!/usr/bin/env python3
"""
Herbal Medicine Chatbot API Server
FastAPI backend for serving the fine-tuned model
"""

import os
import json
import torch
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import uvicorn
import time

# Request/Response models
class ChatRequest(BaseModel):
    message: str
    max_length: Optional[int] = 200
    temperature: Optional[float] = 0.7

class ChatResponse(BaseModel):
    response: str
    confidence: float
    response_time: float
    model_info: Dict[str, Any]

class HerbChatbotAPI:
    def __init__(self, model_path: str, base_model_name: str = "distilgpt2"):
        self.model_path = model_path
        self.base_model_name = base_model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"🚀 Initializing Herbal Medicine Chatbot API")
        print(f"💻 Using device: {self.device}")
        print(f"📁 Model path: {model_path}")
        
        self.load_model()
        
    def load_model(self):
        """Load the fine-tuned model and tokenizer"""
        try:
            print(f"📥 Loading model from {self.model_path}")
            
            # Load tokenizer
            if Path(self.model_path).exists():
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Try to load as fine-tuned model first
            if Path(self.model_path).exists():
                try:
                    # Try loading as a complete fine-tuned model
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                        device_map="auto" if self.device.type == "cuda" else None
                    )
                    self.model_type = "fine_tuned"
                    print("✅ Loaded complete fine-tuned model")
                    
                except Exception as e:
                    print(f"⚠️ Could not load as complete model: {e}")
                    
                    # Try loading as PEFT model
                    try:
                        base_model = AutoModelForCausalLM.from_pretrained(
                            self.base_model_name,
                            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                            device_map="auto" if self.device.type == "cuda" else None
                        )
                        self.model = PeftModel.from_pretrained(base_model, self.model_path)
                        self.model_type = "peft"
                        print("✅ Loaded PEFT model")
                        
                    except Exception as e2:
                        print(f"❌ Could not load PEFT model: {e2}")
                        raise e2
            else:
                # Fallback to base model
                print(f"⚠️ Model path not found, using base model: {self.base_model_name}")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_name,
                    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                    device_map="auto" if self.device.type == "cuda" else None
                )
                self.model_type = "base"
            
            # Set to evaluation mode
            self.model.eval()
            
            # Load model info if available
            info_path = Path(self.model_path) / "training_info.json"
            if info_path.exists():
                with open(info_path, 'r') as f:
                    self.model_info = json.load(f)
            else:
                self.model_info = {
                    "model_name": self.base_model_name,
                    "model_type": self.model_type
                }
            
            print(f"✅ Model loaded successfully")
            print(f"📊 Model parameters: {self.model.num_parameters():,}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise e
    
    def generate_response(self, message: str, max_length: int = 200, temperature: float = 0.7) -> Dict[str, Any]:
        """Generate response to user message"""
        start_time = time.time()
        
        try:
            # Format the prompt for herbal medicine context
            system_prompt = "You are a helpful assistant specializing in herbal medicine. Provide accurate, safe, and informative answers about herbs and their uses."
            formatted_prompt = f"Human: {message}\n\nAssistant:"
            
            # Tokenize input
            inputs = self.tokenizer.encode(formatted_prompt, return_tensors="pt").to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=len(inputs[0]) + max_length,
                    num_return_sequences=1,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.1
                )
            
            # Decode response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the assistant's response
            if "Assistant:" in full_response:
                response = full_response.split("Assistant:")[-1].strip()
            else:
                response = full_response[len(formatted_prompt):].strip()
            
            # Clean up response
            response = response.replace("Human:", "").strip()
            
            # Calculate confidence (simple heuristic based on response length and coherence)
            confidence = min(0.95, max(0.3, len(response) / 200))
            
            response_time = time.time() - start_time
            
            return {
                "response": response,
                "confidence": confidence,
                "response_time": response_time,
                "model_info": self.model_info
            }
            
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            return {
                "response": "I apologize, but I encountered an error while processing your question. Please try rephrasing your question about herbal medicine.",
                "confidence": 0.0,
                "response_time": time.time() - start_time,
                "model_info": self.model_info
            }

# Initialize FastAPI app
app = FastAPI(
    title="Herbal Medicine Chatbot API",
    description="API for querying herbal medicine information using fine-tuned language models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global chatbot instance
chatbot = None

@app.on_event("startup")
async def startup_event():
    """Initialize the chatbot on startup"""
    global chatbot
    
    # Get model path from environment or use default
    model_path = os.getenv("MODEL_PATH", "models/distilgpt2_herbal_final")
    base_model = os.getenv("BASE_MODEL", "distilgpt2")
    
    try:
        chatbot = HerbChatbotAPI(model_path, base_model)
        print("🚀 Chatbot API ready!")
    except Exception as e:
        print(f"❌ Failed to initialize chatbot: {e}")
        # Initialize with base model as fallback
        chatbot = HerbChatbotAPI("", base_model)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Herbal Medicine Chatbot API is running!",
        "status": "healthy",
        "model_info": chatbot.model_info if chatbot else None
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": chatbot is not None,
        "device": str(chatbot.device) if chatbot else None,
        "model_info": chatbot.model_info if chatbot else None
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint"""
    if not chatbot:
        raise HTTPException(status_code=500, detail="Chatbot not initialized")
    
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    try:
        result = chatbot.generate_response(
            request.message,
            request.max_length,
            request.temperature
        )
        
        return ChatResponse(**result)
        
    except Exception as e:
        print(f"❌ Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded model"""
    if not chatbot:
        raise HTTPException(status_code=500, detail="Chatbot not initialized")
    
    return {
        "model_info": chatbot.model_info,
        "model_type": chatbot.model_type,
        "device": str(chatbot.device),
        "parameters": chatbot.model.num_parameters()
    }

def main():
    parser = argparse.ArgumentParser(description="Run Herbal Medicine Chatbot API")
    parser.add_argument("--model_path", default="models/distilgpt2_herbal_final", 
                       help="Path to fine-tuned model")
    parser.add_argument("--base_model", default="distilgpt2", 
                       help="Base model name")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    # Set environment variables
    os.environ["MODEL_PATH"] = args.model_path
    os.environ["BASE_MODEL"] = args.base_model
    
    print(f"🚀 Starting Herbal Medicine Chatbot API")
    print(f"📁 Model path: {args.model_path}")
    print(f"🌐 Server: http://{args.host}:{args.port}")
    print(f"📚 API docs: http://{args.host}:{args.port}/docs")
    
    # Run the server
    uvicorn.run(
        "api:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )

if __name__ == "__main__":
    main()