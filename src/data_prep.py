#!/usr/bin/env python3
"""
Herbal Medicine Dataset Preparation Script
Generates Q&A pairs from herbs dataset for fine-tuning
"""

import json
import pandas as pd
import random
from pathlib import Path
from typing import List, Dict, Any
import argparse

class HerbDataProcessor:
    def __init__(self, input_file: str = "herbs.json"):
        self.input_file = input_file
        self.output_dir = Path("data")
        self.output_dir.mkdir(exist_ok=True)
        
    def load_herbs_data(self) -> List[Dict]:
        """Load herbs dataset from JSON file"""
        try:
            with open(self.input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"✅ Loaded {len(data)} herbs from {self.input_file}")
            return data
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return []
    
    def generate_qa_pairs(self, herb: Dict) -> List[Dict]:
        """Generate 10 Q&A pairs for each herb"""
        qa_pairs = []
        
        # Extract herb info
        scientific_name = herb.get("Scientific Name", "")
        local_name = herb.get("Local Name", "")
        family = herb.get("Family", "")
        uses = herb.get("Uses", "")
        preparation = herb.get("Preparation & Dosage", "")
        side_effects = herb.get("Side Effects / Precautions", "")
        parts_used = herb.get("Parts Used", "")
        location = herb.get("Location", "")
        herb_type = herb.get("Type", "")
        
        # Template questions for each herb
        templates = [
            {
                "question": f"What is {local_name}?",
                "answer": f"{local_name} is the local name for {scientific_name}, a {herb_type.lower()} from the {family} family."
            },
            {
                "question": f"What are the uses of {scientific_name}?",
                "answer": f"{scientific_name} is used for: {uses}"
            },
            {
                "question": f"How do you prepare {local_name}?",
                "answer": f"Preparation of {local_name}: {preparation}"
            },
            {
                "question": f"What are the side effects of {scientific_name}?",
                "answer": f"Side effects and precautions for {scientific_name}: {side_effects}"
            },
            {
                "question": f"Which parts of {local_name} are used medicinally?",
                "answer": f"The medicinal parts of {local_name} ({scientific_name}) are: {parts_used}"
            },
            {
                "question": f"Where is {scientific_name} found?",
                "answer": f"{scientific_name} ({local_name}) is found in: {location}"
            },
            {
                "question": f"What family does {local_name} belong to?",
                "answer": f"{local_name} ({scientific_name}) belongs to the {family} family."
            },
            {
                "question": f"Tell me about the medicinal properties of {scientific_name}",
                "answer": f"{scientific_name} ({local_name}) has the following medicinal properties: {uses}. It is prepared as: {preparation}"
            },
            {
                "question": f"Is {local_name} safe to use?",
                "answer": f"Safety information for {local_name} ({scientific_name}): {side_effects}"
            },
            {
                "question": f"What type of plant is {scientific_name}?",
                "answer": f"{scientific_name} ({local_name}) is a {herb_type.lower()} from the {family} family, found in {location}."
            }
        ]
        
        # Add variations with different question formats
        variations = [
            f"Can you tell me about {local_name}?",
            f"What do you know about {scientific_name}?",
            f"How is {local_name} used in traditional medicine?",
            f"What are the benefits of {scientific_name}?",
            f"How should I use {local_name}?",
        ]
        
        # Add the template Q&A pairs
        for template in templates:
            if template["answer"].strip() and not template["answer"].endswith(": "):
                qa_pairs.append({
                    "prompt": template["question"],
                    "completion": template["answer"]
                })
        
        return qa_pairs
    
    def create_training_format(self, qa_pairs: List[Dict]) -> List[Dict]:
        """Convert Q&A pairs to training format"""
        training_data = []
        
        for qa in qa_pairs:
            # Format for instruction-following models
            formatted_data = {
                "instruction": "You are a helpful assistant specializing in herbal medicine. Answer questions about herbs accurately and safely.",
                "input": qa["prompt"],
                "output": qa["completion"]
            }
            training_data.append(formatted_data)
            
            # Also create a conversational format
            conversation_format = {
                "text": f"Human: {qa['prompt']}\n\nAssistant: {qa['completion']}"
            }
            training_data.append(conversation_format)
        
        return training_data
    
    def save_dataset(self, data: List[Dict], filename: str):
        """Save dataset in JSONL format"""
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"✅ Saved {len(data)} samples to {output_path}")
    
    def process_dataset(self, max_herbs: int = None):
        """Main processing function"""
        print("🌿 Starting Herbal Medicine Dataset Preparation...")
        
        # Load data
        herbs_data = self.load_herbs_data()
        if not herbs_data:
            return
        
        # Limit herbs if specified
        if max_herbs:
            herbs_data = herbs_data[:max_herbs]
            print(f"📊 Processing first {len(herbs_data)} herbs")
        
        # Generate Q&A pairs
        all_qa_pairs = []
        for i, herb in enumerate(herbs_data):
            qa_pairs = self.generate_qa_pairs(herb)
            all_qa_pairs.extend(qa_pairs)
            
            if (i + 1) % 100 == 0:
                print(f"📝 Processed {i + 1}/{len(herbs_data)} herbs...")
        
        print(f"✅ Generated {len(all_qa_pairs)} Q&A pairs from {len(herbs_data)} herbs")
        
        # Create training format
        training_data = self.create_training_format(all_qa_pairs)
        
        # Split into train/validation
        random.shuffle(training_data)
        split_idx = int(0.9 * len(training_data))
        
        train_data = training_data[:split_idx]
        val_data = training_data[split_idx:]
        
        # Save datasets
        self.save_dataset(all_qa_pairs, "qa_pairs.jsonl")
        self.save_dataset(train_data, "train_dataset.jsonl")
        self.save_dataset(val_data, "val_dataset.jsonl")
        
        # Save summary
        summary = {
            "total_herbs": len(herbs_data),
            "total_qa_pairs": len(all_qa_pairs),
            "training_samples": len(train_data),
            "validation_samples": len(val_data),
            "avg_qa_per_herb": len(all_qa_pairs) / len(herbs_data)
        }
        
        with open(self.output_dir / "dataset_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n📊 Dataset Summary:")
        print(f"   Total herbs: {summary['total_herbs']}")
        print(f"   Total Q&A pairs: {summary['total_qa_pairs']}")
        print(f"   Training samples: {summary['training_samples']}")
        print(f"   Validation samples: {summary['validation_samples']}")
        print(f"   Average Q&A per herb: {summary['avg_qa_per_herb']:.1f}")
        
        return summary

def main():
    parser = argparse.ArgumentParser(description="Prepare herbal medicine dataset for training")
    parser.add_argument("--input", default="herbs.json", help="Input herbs JSON file")
    parser.add_argument("--max_herbs", type=int, help="Maximum number of herbs to process")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Process dataset
    processor = HerbDataProcessor(args.input)
    processor.process_dataset(args.max_herbs)

if __name__ == "__main__":
    main()