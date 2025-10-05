#!/usr/bin/env python3
"""
Fine-tuning Script for Herbal Medicine Chatbot
Supports both CPU and GPU training with parameter-efficient methods
"""

import json
import torch
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from torch.nn.utils.rnn import pad_sequence
import torch
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import wandb
from peft import LoraConfig, get_peft_model, TaskType

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HerbalDataset:
    def __init__(self, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def load_jsonl(self, file_path: str) -> List[Dict[str, str]]:
        """Load data from JSONL file"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data
    
    def format_prompt(self, prompt: str, completion: str) -> str:
        """Format prompt and completion for training"""
        return f"Human: {prompt}\nAssistant: {completion}<|endoftext|>"
    
    def tokenize_function(self, examples):
        """Tokenize the examples"""
        # Combine prompt and completion
        texts = [self.format_prompt(p, c) for p, c in zip(examples['prompt'], examples['completion'])]
        
        # Tokenize with proper settings
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            padding=True,  # Enable padding
            max_length=self.max_length,
            return_tensors=None  # Return lists, not tensors
        )
        
        # For causal LM, labels are the same as input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    def prepare_dataset(self, data_path: str) -> Dataset:
        """Prepare dataset for training"""
        data = self.load_jsonl(data_path)
        
        # Convert to the format expected by the tokenizer
        dataset_dict = {
            "prompt": [item["prompt"] for item in data],
            "completion": [item["completion"] for item in data]
        }
        
        dataset = Dataset.from_dict(dataset_dict)
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset

class CustomDataCollator:
    def __init__(self, tokenizer, pad_token_id):
        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id
    
    def __call__(self, features):
        # Extract input_ids and labels
        input_ids = [torch.tensor(f["input_ids"]) for f in features]
        labels = [torch.tensor(f["labels"]) for f in features]
        
        # Pad sequences
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": (input_ids != self.pad_token_id).long()
        }

class HerbalTrainer:
    def __init__(self, model_name: str = "distilgpt2", use_lora: bool = True, device: str = "auto"):
        self.model_name = model_name
        self.use_lora = use_lora
        self.device = device
        self.tokenizer = None
        self.model = None
        
        # Set device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Using device: {self.device}")
        
    def load_model_and_tokenizer(self):
        """Load model and tokenizer"""
        logger.info(f"Loading model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        
        # Apply LoRA if requested
        if self.use_lora:
            logger.info("Applying LoRA configuration")
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=16,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["c_attn"] if "distilgpt2" in self.model_name else ["q_proj", "v_proj"]
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
        
        logger.info("Model and tokenizer loaded successfully")
    
    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=-1)
        
        # Remove padding tokens (-100)
        true_predictions = []
        true_labels = []
        
        for prediction, label in zip(predictions, labels):
            for pred, lab in zip(prediction, label):
                if lab != -100:
                    true_predictions.append(pred)
                    true_labels.append(lab)
        
        if len(true_predictions) == 0:
            return {"accuracy": 0.0}
        
        accuracy = accuracy_score(true_labels, true_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, true_predictions, average='weighted'
        )
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    
    def train(self, train_dataset, eval_dataset=None, output_dir: str = "herbal_model", 
              num_epochs: int = 3, batch_size: int = 4, learning_rate: float = 5e-5,
              warmup_steps: int = 100, logging_steps: int = 10, eval_steps: int = 100,
              save_steps: int = 500, use_wandb: bool = False):
        """Train the model"""
        
        # Initialize wandb if requested
        if use_wandb:
            wandb.init(
                project="herbal-medicine-chatbot",
                config={
                    "model_name": self.model_name,
                    "use_lora": self.use_lora,
                    "num_epochs": num_epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate
                }
            )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=warmup_steps,
            learning_rate=learning_rate,
            logging_steps=logging_steps,
            eval_steps=eval_steps,
            save_steps=save_steps,
            eval_strategy="steps" if eval_dataset else "no",
            save_strategy="steps",
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="wandb" if use_wandb else None,
            remove_unused_columns=False,
            dataloader_pin_memory=False if self.device == "cpu" else True,
            fp16=self.device == "cuda",
            gradient_accumulation_steps=4 if self.device == "cpu" else 1,
        )
        
        # Data collator
        data_collator = CustomDataCollator(
            tokenizer=self.tokenizer,
            pad_token_id=self.tokenizer.pad_token_id
        )
        
        # Callbacks
        callbacks = []
        if eval_dataset:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics if eval_dataset else None,
            callbacks=callbacks,
        )
        
        # Train
        logger.info("Starting training...")
        trainer.train()
        
        # Save model
        logger.info(f"Saving model to {output_dir}")
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        # Save training info
        training_info = {
            "model_name": self.model_name,
            "use_lora": self.use_lora,
            "device": self.device,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "train_samples": len(train_dataset),
            "eval_samples": len(eval_dataset) if eval_dataset else 0,
            "training_time": datetime.now().isoformat()
        }
        
        with open(f"{output_dir}/training_info.json", "w") as f:
            json.dump(training_info, f, indent=2)
        
        logger.info("Training completed successfully!")
        
        if use_wandb:
            wandb.finish()
        
        return trainer

def main():
    parser = argparse.ArgumentParser(description="Fine-tune herbal medicine chatbot")
    parser.add_argument("--model", default="distilgpt2", help="Base model name")
    parser.add_argument("--train-data", default="data/train.jsonl", help="Training data path")
    parser.add_argument("--eval-data", default="data/test.jsonl", help="Evaluation data path")
    parser.add_argument("--output-dir", default="herbal_model", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--use-lora", action="store_true", help="Use LoRA for parameter-efficient training")
    parser.add_argument("--device", default="auto", help="Device to use (auto/cpu/cuda)")
    parser.add_argument("--use-wandb", action="store_true", help="Use Weights & Biases for logging")
    parser.add_argument("--subset", type=int, help="Use only first N examples for quick testing")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = HerbalTrainer(
        model_name=args.model,
        use_lora=args.use_lora,
        device=args.device
    )
    
    # Load model and tokenizer
    trainer.load_model_and_tokenizer()
    
    # Prepare dataset
    dataset_handler = HerbalDataset(trainer.tokenizer, args.max_length)
    
    logger.info("Loading training dataset...")
    train_dataset = dataset_handler.prepare_dataset(args.train_data)
    
    eval_dataset = None
    if Path(args.eval_data).exists():
        logger.info("Loading evaluation dataset...")
        eval_dataset = dataset_handler.prepare_dataset(args.eval_data)
    
    # Use subset if specified
    if args.subset:
        train_dataset = train_dataset.select(range(min(args.subset, len(train_dataset))))
        if eval_dataset:
            eval_dataset = eval_dataset.select(range(min(args.subset // 4, len(eval_dataset))))
        logger.info(f"Using subset: {len(train_dataset)} train, {len(eval_dataset) if eval_dataset else 0} eval")
    
    # Train
    trainer.train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_wandb=args.use_wandb
    )

if __name__ == "__main__":
    main()
