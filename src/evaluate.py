#!/usr/bin/env python3
"""
Herbal Medicine Chatbot Evaluation Script
Evaluates fine-tuned model performance vs base model
"""

import json
import torch
import argparse
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import time

class HerbChatbotEvaluator:
    def __init__(self, base_model_name: str, fine_tuned_path: str = None):
        self.base_model_name = base_model_name
        self.fine_tuned_path = fine_tuned_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"🔍 Initializing evaluator")
        print(f"💻 Using device: {self.device}")
        
    def load_models(self):
        """Load base and fine-tuned models"""
        print(f"📥 Loading base model: {self.base_model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            device_map="auto" if self.device.type == "cuda" else None
        )
        
        # Load fine-tuned model if provided
        if self.fine_tuned_path:
            print(f"📥 Loading fine-tuned model: {self.fine_tuned_path}")
            
            # Check if it's a PEFT model
            if Path(self.fine_tuned_path).exists():
                try:
                    self.fine_tuned_model = PeftModel.from_pretrained(
                        self.base_model, 
                        self.fine_tuned_path
                    )
                except:
                    # If not PEFT, load as regular model
                    self.fine_tuned_model = AutoModelForCausalLM.from_pretrained(
                        self.fine_tuned_path,
                        torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                        device_map="auto" if self.device.type == "cuda" else None
                    )
            else:
                print(f"❌ Fine-tuned model path not found: {self.fine_tuned_path}")
                self.fine_tuned_model = None
        else:
            self.fine_tuned_model = None
        
        print("✅ Models loaded successfully")
    
    def generate_response(self, model, prompt: str, max_length: int = 200) -> str:
        """Generate response from model"""
        # Format prompt
        formatted_prompt = f"Human: {prompt}\n\nAssistant:"
        
        # Tokenize
        inputs = self.tokenizer.encode(formatted_prompt, return_tensors="pt").to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=len(inputs[0]) + max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=3
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant's response
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()
        
        return response
    
    def calculate_perplexity(self, model, dataset, max_samples: int = 100):
        """Calculate perplexity on validation dataset"""
        print(f"📊 Calculating perplexity on {max_samples} samples...")
        
        model.eval()
        total_loss = 0
        total_tokens = 0
        
        samples = dataset.select(range(min(max_samples, len(dataset))))
        
        for sample in tqdm(samples):
            if "text" in sample:
                text = sample["text"]
            else:
                text = f"Human: {sample['input']}\n\nAssistant: {sample['output']}"
            
            # Tokenize
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Calculate loss
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                
                total_loss += loss.item() * inputs["input_ids"].size(1)
                total_tokens += inputs["input_ids"].size(1)
        
        perplexity = torch.exp(torch.tensor(total_loss / total_tokens))
        return perplexity.item()
    
    def evaluate_responses(self, test_questions: list, max_samples: int = 20):
        """Evaluate response quality on test questions"""
        print(f"🧪 Evaluating responses on {len(test_questions)} questions...")
        
        results = {
            "base_model": [],
            "fine_tuned_model": [] if self.fine_tuned_model else None
        }
        
        for i, question in enumerate(test_questions[:max_samples]):
            print(f"\n📝 Question {i+1}: {question}")
            
            # Base model response
            start_time = time.time()
            base_response = self.generate_response(self.base_model, question)
            base_time = time.time() - start_time
            
            print(f"🤖 Base model: {base_response[:100]}...")
            
            results["base_model"].append({
                "question": question,
                "response": base_response,
                "response_time": base_time
            })
            
            # Fine-tuned model response
            if self.fine_tuned_model:
                start_time = time.time()
                ft_response = self.generate_response(self.fine_tuned_model, question)
                ft_time = time.time() - start_time
                
                print(f"🎯 Fine-tuned: {ft_response[:100]}...")
                
                results["fine_tuned_model"].append({
                    "question": question,
                    "response": ft_response,
                    "response_time": ft_time
                })
        
        return results
    
    def run_evaluation(self, val_dataset_path: str = "data/val_dataset.jsonl"):
        """Run complete evaluation"""
        print("🚀 Starting comprehensive evaluation...")
        
        # Load validation dataset
        val_dataset = load_dataset("json", data_files=val_dataset_path, split="train")
        
        results = {}
        
        # Calculate perplexity
        print("\n📊 Calculating perplexity...")
        base_perplexity = self.calculate_perplexity(self.base_model, val_dataset)
        results["base_perplexity"] = base_perplexity
        print(f"Base model perplexity: {base_perplexity:.2f}")
        
        if self.fine_tuned_model:
            ft_perplexity = self.calculate_perplexity(self.fine_tuned_model, val_dataset)
            results["fine_tuned_perplexity"] = ft_perplexity
            print(f"Fine-tuned model perplexity: {ft_perplexity:.2f}")
            print(f"Perplexity improvement: {((base_perplexity - ft_perplexity) / base_perplexity * 100):.1f}%")
        
        # Test questions for qualitative evaluation
        test_questions = [
            "What are the uses of Sokhrus?",
            "How do you prepare Chamomile?",
            "What are the side effects of Astragalus?",
            "Which parts of Equisetum are used medicinally?",
            "Tell me about Rhodiola imbricata",
            "What family does Capparis spinosa belong to?",
            "How is Tussilago farfara prepared?",
            "What are the benefits of Swertia cordata?",
            "Is Nepeta erecta safe to use?",
            "Where is Primula macrophylla found?"
        ]
        
        # Evaluate responses
        response_results = self.evaluate_responses(test_questions)
        results["response_evaluation"] = response_results
        
        # Calculate average response times
        base_avg_time = np.mean([r["response_time"] for r in response_results["base_model"]])
        results["base_avg_response_time"] = base_avg_time
        
        if response_results["fine_tuned_model"]:
            ft_avg_time = np.mean([r["response_time"] for r in response_results["fine_tuned_model"]])
            results["fine_tuned_avg_response_time"] = ft_avg_time
        
        # Save results
        output_path = Path("models") / "evaluation_results.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n✅ Evaluation completed!")
        print(f"📁 Results saved to: {output_path}")
        
        # Print summary
        print(f"\n📊 Evaluation Summary:")
        print(f"   Base model perplexity: {base_perplexity:.2f}")
        if self.fine_tuned_model:
            print(f"   Fine-tuned perplexity: {ft_perplexity:.2f}")
            print(f"   Improvement: {((base_perplexity - ft_perplexity) / base_perplexity * 100):.1f}%")
        print(f"   Average response time: {base_avg_time:.2f}s")
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate herbal medicine chatbot")
    parser.add_argument("--base_model", default="distilgpt2", help="Base model name")
    parser.add_argument("--fine_tuned_path", help="Path to fine-tuned model")
    parser.add_argument("--val_dataset", default="data/val_dataset.jsonl", help="Validation dataset path")
    parser.add_argument("--max_samples", type=int, default=100, help="Maximum samples for evaluation")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = HerbChatbotEvaluator(args.base_model, args.fine_tuned_path)
    
    # Load models
    evaluator.load_models()
    
    # Run evaluation
    results = evaluator.run_evaluation(args.val_dataset)
    
    print("\n🎉 Evaluation completed successfully!")

if __name__ == "__main__":
    main()