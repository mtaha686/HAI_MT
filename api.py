#!/usr/bin/env python3
"""
FastAPI Backend for Herbal Medicine Chatbot
Serves the fine-tuned model via REST API
"""

import json
import torch
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import uvicorn
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import asyncio

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model and tokenizer
model = None
tokenizer = None
model_info = None

class ChatRequest(BaseModel):
    message: str = Field(..., description="User's question about herbs")
    max_length: int = Field(default=200, description="Maximum response length")
    temperature: float = Field(default=0.7, description="Sampling temperature")
    top_p: float = Field(default=0.9, description="Top-p sampling parameter")

class ChatResponse(BaseModel):
    response: str = Field(..., description="Bot's response")
    timestamp: str = Field(..., description="Response timestamp")
    model_info: Dict[str, Any] = Field(..., description="Model information")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    timestamp: str = Field(..., description="Health check timestamp")

class HerbalChatbot:
    def __init__(self, model_path: str, base_model_name: str = "distilgpt2"):
        self.model_path = model_path
        self.base_model_name = base_model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.model_info = None
        
        logger.info(f"Using device: {self.device}")
    
    def load_model(self):
        """Load the fine-tuned model and tokenizer"""
        try:
            logger.info("Loading tokenizer...")
            if self.model_path and Path(self.model_path).exists():
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            else:
                logger.info("Loading base model tokenizer...")
                self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info("Loading model...")
            # Check if it's a LoRA model
            if self.model_path and Path(self.model_path).exists() and (Path(self.model_path) / "adapter_config.json").exists():
                logger.info("Loading LoRA fine-tuned model...")
                base_model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None
                )
                self.model = PeftModel.from_pretrained(base_model, self.model_path)
            elif self.model_path and Path(self.model_path).exists():
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None
                )
            else:
                logger.info("Loading base model...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None
                )
            
            # Load model info
            if self.model_path and Path(self.model_path).exists():
                info_path = Path(self.model_path) / "training_info.json"
                if info_path.exists():
                    with open(info_path, 'r') as f:
                        self.model_info = json.load(f)
                else:
                    self.model_info = {
                        "model_name": self.base_model_name,
                        "device": self.device,
                        "loaded_at": datetime.now().isoformat()
                    }
            else:
                self.model_info = {
                    "model_name": self.base_model_name,
                    "model_type": "base_model",
                    "device": self.device,
                    "loaded_at": datetime.now().isoformat()
                }
            
            logger.info("Model loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def format_prompt(self, message: str) -> str:
        """Format user message for the model"""
        return f"Human: {message}\nAssistant: To answer your question about herbal medicine:"
    
    def generate_response(self, message: str, max_length: int = 200, 
                         temperature: float = 0.7, top_p: float = 0.9) -> str:
        """Generate response from the model"""
        try:
            formatted_prompt = self.format_prompt(message)
            
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=inputs['input_ids'].shape[1] + max_length,
                    temperature=0.3,  # Lower temperature for more focused responses
                    top_p=0.8,        # More focused sampling
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.2,  # Higher repetition penalty
                    num_beams=2       # Use beam search for better quality
                )
            
            # Decode only the generated part
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )
            
            return generated_text.strip()
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

# Initialize chatbot
chatbot = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    global chatbot, model, tokenizer, model_info
    
    # Startup
    logger.info("Starting Herbal Medicine Chatbot API...")
    
    # Initialize chatbot
    model_path = "herbal_model"  # Default model path
    if not Path(model_path).exists():
        logger.warning(f"Model path {model_path} does not exist. Using base model as fallback.")
        # Use base model as fallback
        chatbot = HerbalChatbot("", "distilgpt2")  # Empty path will use base model
    else:
        chatbot = HerbalChatbot(model_path)
    
    if chatbot.load_model():
        model = chatbot.model
        tokenizer = chatbot.tokenizer
        model_info = chatbot.model_info
        logger.info("API ready to serve requests!")
    else:
        logger.error("Failed to load model. API will not function properly.")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Herbal Medicine Chatbot API...")

# Create FastAPI app
app = FastAPI(
    title="Herbal Medicine Chatbot API",
    description="API for the Herbal Medicine Chatbot powered by fine-tuned language models",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Herbal Medicine Chatbot API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        timestamp=datetime.now().isoformat()
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint"""
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please check the health endpoint."
        )
    
    try:
        # Generate response
        response_text = chatbot.generate_response(
            message=request.message,
            max_length=request.max_length,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        return ChatResponse(
            response=response_text,
            timestamp=datetime.now().isoformat(),
            model_info=model_info
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info", response_model=Dict[str, Any])
async def get_model_info():
    """Get model information"""
    if model_info is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return model_info

@app.get("/examples", response_model=List[str])
async def get_example_questions():
    """Get example questions users can ask"""
    examples = [
        "What is the scientific name of Sokhrus?",
        "What are the medicinal uses of Sokhrus?",
        "How do I prepare Sokhrus?",
        "What are the side effects of Sokhrus?",
        "Tell me about Sokhrus herb",
        "Where can I find Sokhrus?",
        "What family does Sokhrus belong to?",
        "What parts of Sokhrus are used medicinally?",
        "Is Sokhrus safe to use?",
        "What type of plant is Sokhrus?"
    ]
    return examples

@app.post("/batch-chat", response_model=List[ChatResponse])
async def batch_chat(requests: List[ChatRequest]):
    """Process multiple chat requests in batch"""
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please check the health endpoint."
        )
    
    responses = []
    for request in requests:
        try:
            response_text = chatbot.generate_response(
                message=request.message,
                max_length=request.max_length,
                temperature=request.temperature,
                top_p=request.top_p
            )
            
            responses.append(ChatResponse(
                response=response_text,
                timestamp=datetime.now().isoformat(),
                model_info=model_info
            ))
            
        except Exception as e:
            logger.error(f"Error in batch chat for message '{request.message}': {e}")
            responses.append(ChatResponse(
                response=f"Error processing request: {str(e)}",
                timestamp=datetime.now().isoformat(),
                model_info=model_info
            ))
    
    return responses

@app.get("/stats", response_model=Dict[str, Any])
async def get_stats():
    """Get API statistics"""
    return {
        "model_loaded": model is not None,
        "device": chatbot.device if chatbot else "unknown",
        "model_path": chatbot.model_path if chatbot else "unknown",
        "base_model": chatbot.base_model_name if chatbot else "unknown",
        "uptime": datetime.now().isoformat()
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "Endpoint not found"}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

def main():
    """Run the API server"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Herbal Medicine Chatbot API")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    args = parser.parse_args()
    
    logger.info(f"Starting server on {args.host}:{args.port}")
    
    uvicorn.run(
        "api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )

if __name__ == "__main__":
    main()
