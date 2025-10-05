#!/usr/bin/env python3
"""
Evaluation Script for Herbal Medicine Chatbot
Provides comprehensive evaluation metrics and comparison with base model
"""

import json
import torch
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
from datetime import datetime

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from peft import PeftModel
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HerbalEvaluator:
    def __init__(self, model_path: str, base_model_name: str = "distilgpt2"):
        self.model_path = model_path
        self.base_model_name = base_model_name
        self.tokenizer = None
        self.fine_tuned_model = None
        self.base_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Using device: {self.device}")
    
    def load_models(self):
        """Load both fine-tuned and base models"""
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info("Loading fine-tuned model...")
        # Check if it's a LoRA model
        if (Path(self.model_path) / "adapter_config.json").exists():
            logger.info("Loading LoRA fine-tuned model...")
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            self.fine_tuned_model = PeftModel.from_pretrained(base_model, self.model_path)
        else:
            self.fine_tuned_model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
        
        logger.info("Loading base model for comparison...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        
        logger.info("Models loaded successfully")
    
    def load_test_data(self, test_data_path: str) -> List[Dict[str, str]]:
        """Load test data from JSONL file"""
        data = []
        with open(test_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data
    
    def format_prompt(self, prompt: str) -> str:
        """Format prompt for inference"""
        return f"Human: {prompt}\nAssistant:"
    
    def generate_response(self, model, prompt: str, max_length: int = 200, 
                         temperature: float = 0.7, top_p: float = 0.9) -> str:
        """Generate response from model"""
        formatted_prompt = self.format_prompt(prompt)
        
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=inputs['input_ids'].shape[1] + max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode only the generated part
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        
        return generated_text.strip()
    
    def evaluate_perplexity(self, model, test_data: List[Dict[str, str]]) -> float:
        """Calculate perplexity on test data"""
        logger.info("Calculating perplexity...")
        
        total_loss = 0
        total_tokens = 0
        
        for item in test_data[:100]:  # Use subset for efficiency
            prompt = item["prompt"]
            completion = item["completion"]
            text = f"Human: {prompt}\nAssistant: {completion}<|endoftext|>"
            
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                total_loss += loss.item() * inputs["input_ids"].shape[1]
                total_tokens += inputs["input_ids"].shape[1]
        
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return perplexity
    
    def evaluate_quality(self, test_data: List[Dict[str, str]], 
                        num_samples: int = 50) -> Dict[str, Any]:
        """Evaluate response quality"""
        logger.info(f"Evaluating quality on {num_samples} samples...")
        
        results = {
            "fine_tuned": [],
            "base": []
        }
        
        sample_data = test_data[:num_samples]
        
        for i, item in enumerate(sample_data):
            if i % 10 == 0:
                logger.info(f"Evaluating sample {i+1}/{len(sample_data)}")
            
            prompt = item["prompt"]
            expected = item["completion"]
            
            # Generate responses
            ft_response = self.generate_response(self.fine_tuned_model, prompt)
            base_response = self.generate_response(self.base_model, prompt)
            
            results["fine_tuned"].append({
                "prompt": prompt,
                "expected": expected,
                "generated": ft_response,
                "length": len(ft_response.split())
            })
            
            results["base"].append({
                "prompt": prompt,
                "expected": expected,
                "generated": base_response,
                "length": len(base_response.split())
            })
        
        return results
    
    def calculate_metrics(self, results: Dict[str, List[Dict]]) -> Dict[str, Dict[str, float]]:
        """Calculate evaluation metrics"""
        metrics = {}
        
        for model_type, responses in results.items():
            avg_length = np.mean([r["length"] for r in responses])
            
            # Simple keyword-based relevance scoring
            relevance_scores = []
            for r in responses:
                expected_words = set(r["expected"].lower().split())
                generated_words = set(r["generated"].lower().split())
                
                if len(expected_words) > 0:
                    overlap = len(expected_words.intersection(generated_words))
                    relevance = overlap / len(expected_words)
                    relevance_scores.append(relevance)
                else:
                    relevance_scores.append(0.0)
            
            avg_relevance = np.mean(relevance_scores)
            
            metrics[model_type] = {
                "avg_response_length": avg_length,
                "avg_relevance_score": avg_relevance,
                "num_samples": len(responses)
            }
        
        return metrics
    
    def generate_comparison_report(self, test_data: List[Dict[str, str]], 
                                 output_dir: str = "evaluation_results") -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        logger.info("Starting comprehensive evaluation...")
        
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        # Calculate perplexity
        ft_perplexity = self.evaluate_perplexity(self.fine_tuned_model, test_data)
        base_perplexity = self.evaluate_perplexity(self.base_model, test_data)
        
        # Evaluate quality
        quality_results = self.evaluate_quality(test_data)
        
        # Calculate metrics
        metrics = self.calculate_metrics(quality_results)
        
        # Compile report
        report = {
            "evaluation_info": {
                "timestamp": datetime.now().isoformat(),
                "model_path": self.model_path,
                "base_model": self.base_model_name,
                "test_samples": len(test_data),
                "device": self.device
            },
            "perplexity": {
                "fine_tuned": ft_perplexity,
                "base": base_perplexity,
                "improvement": (base_perplexity - ft_perplexity) / base_perplexity * 100
            },
            "quality_metrics": metrics,
            "sample_responses": quality_results
        }
        
        # Save detailed results
        with open(f"{output_dir}/evaluation_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        # Save sample responses for manual inspection
        sample_responses_df = pd.DataFrame([
            {
                "prompt": r["prompt"],
                "expected": r["expected"],
                "fine_tuned": r["generated"],
                "base": base_r["generated"]
            }
            for r, base_r in zip(quality_results["fine_tuned"], quality_results["base"])
        ])
        
        sample_responses_df.to_csv(f"{output_dir}/sample_responses.csv", index=False)
        
        # Generate summary
        self.generate_summary_report(report, output_dir)
        
        logger.info(f"Evaluation complete! Results saved to {output_dir}")
        
        return report
    
    def generate_summary_report(self, report: Dict[str, Any], output_dir: str):
        """Generate a human-readable summary report"""
        summary = f"""
# Herbal Medicine Chatbot Evaluation Report

## Evaluation Summary
- **Timestamp**: {report['evaluation_info']['timestamp']}
- **Model**: {report['evaluation_info']['model_path']}
- **Base Model**: {report['evaluation_info']['base_model']}
- **Test Samples**: {report['evaluation_info']['test_samples']}
- **Device**: {report['evaluation_info']['device']}

## Perplexity Results
- **Fine-tuned Model**: {report['perplexity']['fine_tuned']:.2f}
- **Base Model**: {report['perplexity']['base']:.2f}
- **Improvement**: {report['perplexity']['improvement']:.1f}%

## Quality Metrics
### Fine-tuned Model
- **Average Response Length**: {report['quality_metrics']['fine_tuned']['avg_response_length']:.1f} words
- **Average Relevance Score**: {report['quality_metrics']['fine_tuned']['avg_relevance_score']:.3f}
- **Samples Evaluated**: {report['quality_metrics']['fine_tuned']['num_samples']}

### Base Model
- **Average Response Length**: {report['quality_metrics']['base']['avg_response_length']:.1f} words
- **Average Relevance Score**: {report['quality_metrics']['base']['avg_relevance_score']:.3f}
- **Samples Evaluated**: {report['quality_metrics']['base']['num_samples']}

## Key Findings
1. **Perplexity**: {'Improved' if report['perplexity']['improvement'] > 0 else 'No improvement'} by {abs(report['perplexity']['improvement']):.1f}%
2. **Relevance**: Fine-tuned model {'outperforms' if report['quality_metrics']['fine_tuned']['avg_relevance_score'] > report['quality_metrics']['base']['avg_relevance_score'] else 'underperforms'} base model
3. **Response Length**: Fine-tuned model generates {'longer' if report['quality_metrics']['fine_tuned']['avg_response_length'] > report['quality_metrics']['base']['avg_response_length'] else 'shorter'} responses

## Recommendations
- {'The fine-tuning was successful!' if report['perplexity']['improvement'] > 0 and report['quality_metrics']['fine_tuned']['avg_relevance_score'] > report['quality_metrics']['base']['avg_relevance_score'] else 'Consider adjusting training parameters or increasing training data.'}
- Review sample responses in `sample_responses.csv` for qualitative assessment
- Consider additional evaluation metrics for production deployment
"""
        
        with open(f"{output_dir}/summary_report.md", "w") as f:
            f.write(summary)
        
        print(summary)

def main():
    parser = argparse.ArgumentParser(description="Evaluate herbal medicine chatbot")
    parser.add_argument("--model-path", required=True, help="Path to fine-tuned model")
    parser.add_argument("--base-model", default="distilgpt2", help="Base model name")
    parser.add_argument("--test-data", default="data/test.jsonl", help="Test data path")
    parser.add_argument("--output-dir", default="evaluation_results", help="Output directory")
    parser.add_argument("--num-samples", type=int, default=50, help="Number of samples for quality evaluation")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = HerbalEvaluator(args.model_path, args.base_model)
    
    # Load models
    evaluator.load_models()
    
    # Load test data
    test_data = evaluator.load_test_data(args.test_data)
    logger.info(f"Loaded {len(test_data)} test samples")
    
    # Generate evaluation report
    report = evaluator.generate_comparison_report(
        test_data, 
        args.output_dir
    )
    
    print("\n" + "="*50)
    print("EVALUATION COMPLETE")
    print("="*50)
    print(f"Results saved to: {args.output_dir}")
    print(f"Perplexity improvement: {report['perplexity']['improvement']:.1f}%")
    print(f"Relevance score (FT): {report['quality_metrics']['fine_tuned']['avg_relevance_score']:.3f}")
    print(f"Relevance score (Base): {report['quality_metrics']['base']['avg_relevance_score']:.3f}")

if __name__ == "__main__":
    main()
