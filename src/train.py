#!/usr/bin/env python3
"""
Herbal Medicine Chatbot Training Script
Fine-tunes language models using LoRA for memory efficiency
"""

import os
import json
import torch
import argparse
from pathlib import Path
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np

class HerbChatbotTrainer:
    def __init__(self, model_name: str = "distilgpt2", max_length: int = 512):
        self.model_name = model_name
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"🚀 Initializing trainer with {model_name}")
        print(f"💻 Using device: {self.device}")
        
        # Create directories
        self.output_dir = Path("models")
        self.output_dir.mkdir(exist_ok=True)
        
    def load_model_and_tokenizer(self):
        """Load pre-trained model and tokenizer"""
        print(f"📥 Loading model and tokenizer: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            device_map="auto" if self.device.type == "cuda" else None
        )
        
        # Resize token embeddings if needed
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        print(f"✅ Model loaded with {self.model.num_parameters():,} parameters")
        
    def setup_lora(self, r: int = 16, alpha: int = 32):
        """Setup LoRA for parameter-efficient fine-tuning"""
        print(f"🔧 Setting up LoRA (r={r}, alpha={alpha})")
        
        # LoRA configuration
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=r,
            lora_alpha=alpha,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"] if "gpt" in self.model_name.lower() else ["c_attn"]
        )
        
        # Apply LoRA to model
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        print(f"📊 Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
        
    def load_dataset(self, train_file: str = "data/train_dataset.jsonl", 
                    val_file: str = "data/val_dataset.jsonl"):
        """Load and preprocess training dataset"""
        print(f"📚 Loading datasets...")
        
        # Load datasets
        train_dataset = load_dataset("json", data_files=train_file, split="train")
        val_dataset = load_dataset("json", data_files=val_file, split="train")
        
        print(f"📊 Train samples: {len(train_dataset)}")
        print(f"📊 Validation samples: {len(val_dataset)}")
        
        # Tokenize datasets
        def tokenize_function(examples):
            # Handle different formats
            if "text" in examples:
                texts = examples["text"]
            else:
                # Combine instruction, input, and output
                texts = []
                for i in range(len(examples["instruction"])):
                    instruction = examples["instruction"][i]
                    input_text = examples["input"][i]
                    output = examples["output"][i]
                    
                    # Format as conversation
                    text = f"Human: {input_text}\n\nAssistant: {output}"
                    texts.append(text)
            
            # Tokenize
            tokenized = self.tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            # For causal LM, labels are the same as input_ids
            tokenized["labels"] = tokenized["input_ids"].clone()
            
            return tokenized
        
        # Apply tokenization
        train_dataset = train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=train_dataset.column_names
        )
        
        val_dataset = val_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=val_dataset.column_names
        )
        
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        return train_dataset, val_dataset
    
    def train(self, 
              num_epochs: int = 3,
              batch_size: int = 4,
              learning_rate: float = 5e-5,
              warmup_steps: int = 100,
              save_steps: int = 500):
        """Train the model"""
        
        print(f"🏋️ Starting training...")
        print(f"   Epochs: {num_epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Learning rate: {learning_rate}")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir / f"{self.model_name.replace('/', '_')}_herbal_chatbot"),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=warmup_steps,
            weight_decay=0.01,
            logging_dir=str(self.output_dir / "logs"),
            logging_steps=50,
            save_steps=save_steps,
            eval_steps=save_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=None,  # Disable wandb/tensorboard
            dataloader_pin_memory=False,
            gradient_accumulation_steps=2,
            fp16=self.device.type == "cuda",
            learning_rate=learning_rate,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # We're doing causal LM, not masked LM
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=data_collator,
        )
        
        # Train
        print("🚀 Training started...")
        trainer.train()
        
        # Save final model
        final_model_path = self.output_dir / f"{self.model_name.replace('/', '_')}_herbal_final"
        trainer.save_model(str(final_model_path))
        self.tokenizer.save_pretrained(str(final_model_path))
        
        print(f"✅ Training completed! Model saved to {final_model_path}")
        
        # Save training info
        training_info = {
            "model_name": self.model_name,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "train_samples": len(self.train_dataset),
            "val_samples": len(self.val_dataset),
            "device": str(self.device),
            "model_path": str(final_model_path)
        }
        
        with open(final_model_path / "training_info.json", 'w') as f:
            json.dump(training_info, f, indent=2)
        
        return trainer, final_model_path

def main():
    parser = argparse.ArgumentParser(description="Train herbal medicine chatbot")
    parser.add_argument("--model_name", default="distilgpt2", 
                       help="Base model name (distilgpt2, microsoft/DialoGPT-medium, etc.)")
    parser.add_argument("--max_samples", type=int, help="Maximum training samples")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = HerbChatbotTrainer(args.model_name, args.max_length)
    
    # Load model and setup LoRA
    trainer.load_model_and_tokenizer()
    trainer.setup_lora(args.lora_r, args.lora_alpha)
    
    # Load dataset
    train_dataset, val_dataset = trainer.load_dataset()
    
    # Limit samples if specified
    if args.max_samples:
        trainer.train_dataset = trainer.train_dataset.select(range(min(args.max_samples, len(trainer.train_dataset))))
        trainer.val_dataset = trainer.val_dataset.select(range(min(args.max_samples // 10, len(trainer.val_dataset))))
        print(f"📊 Limited to {len(trainer.train_dataset)} train and {len(trainer.val_dataset)} val samples")
    
    # Train model
    trained_model, model_path = trainer.train(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    print(f"\n🎉 Training completed successfully!")
    print(f"📁 Model saved at: {model_path}")
    print(f"🚀 Ready to deploy with: python src/api.py --model_path {model_path}")

if __name__ == "__main__":
    main()