#!/usr/bin/env python3
"""
Data Preparation Script for Herbal Medicine Chatbot
Generates Q&A pairs from herbs.json and saves in JSONL format for fine-tuning
"""

import json
import random
from typing import List, Dict, Any
import argparse
from pathlib import Path

class HerbalQAGenerator:
    def __init__(self, herbs_data: List[Dict[str, Any]]):
        self.herbs_data = herbs_data
        self.qa_templates = self._create_qa_templates()
    
    def _create_qa_templates(self) -> List[Dict[str, str]]:
        """Create diverse Q&A templates for each herb"""
        return [
            {
                "question_template": "What is the scientific name of {local_name}?",
                "answer_template": "The scientific name of {local_name} is {scientific_name}."
            },
            {
                "question_template": "What family does {local_name} belong to?",
                "answer_template": "{local_name} belongs to the {family} family."
            },
            {
                "question_template": "Where can I find {local_name}?",
                "answer_template": "{local_name} can be found in {location}."
            },
            {
                "question_template": "What parts of {local_name} are used medicinally?",
                "answer_template": "The medicinal parts of {local_name} include: {parts_used}."
            },
            {
                "question_template": "What are the medicinal uses of {local_name}?",
                "answer_template": "{local_name} has the following medicinal uses: {uses}."
            },
            {
                "question_template": "How do I prepare and use {local_name}?",
                "answer_template": "To prepare and use {local_name}: {preparation_dosage}."
            },
            {
                "question_template": "What are the side effects of {local_name}?",
                "answer_template": "Side effects and precautions for {local_name}: {side_effects}."
            },
            {
                "question_template": "Tell me about {local_name} herb.",
                "answer_template": "{local_name} ({scientific_name}) is a {type} from the {family} family. It can be found in {location}. The medicinal parts used are: {parts_used}. Uses include: {uses}. Preparation: {preparation_dosage}. Side effects: {side_effects}."
            },
            {
                "question_template": "Is {local_name} safe to use?",
                "answer_template": "{local_name} safety information: {side_effects}. Always consult a healthcare provider before use."
            },
            {
                "question_template": "What type of plant is {local_name}?",
                "answer_template": "{local_name} is a {type} plant from the {family} family."
            }
        ]
    
    def generate_qa_pairs(self, num_pairs_per_herb: int = 10) -> List[Dict[str, str]]:
        """Generate Q&A pairs for all herbs"""
        all_qa_pairs = []
        
        for herb in self.herbs_data:
            # Get primary local name (first one if multiple)
            local_name = herb.get("Local Name", "").split(",")[0].strip()
            if not local_name:
                continue
            
            # Generate Q&A pairs for this herb
            herb_qa_pairs = []
            
            # Use all templates
            for template in self.qa_templates:
                try:
                    question = template["question_template"].format(
                        local_name=local_name,
                        scientific_name=herb.get("Scientific Name", "Unknown"),
                        family=herb.get("Family", "Unknown"),
                        location=herb.get("Location", "Unknown"),
                        parts_used=herb.get("Parts Used", "Unknown"),
                        uses=herb.get("Uses", "Unknown"),
                        preparation_dosage=herb.get("Preparation & Dosage", "Unknown"),
                        side_effects=herb.get("Side Effects / Precautions", "Unknown"),
                        type=herb.get("Type", "Unknown")
                    )
                    
                    answer = template["answer_template"].format(
                        local_name=local_name,
                        scientific_name=herb.get("Scientific Name", "Unknown"),
                        family=herb.get("Family", "Unknown"),
                        location=herb.get("Location", "Unknown"),
                        parts_used=herb.get("Parts Used", "Unknown"),
                        uses=herb.get("Uses", "Unknown"),
                        preparation_dosage=herb.get("Preparation & Dosage", "Unknown"),
                        side_effects=herb.get("Side Effects / Precautions", "Unknown"),
                        type=herb.get("Type", "Unknown")
                    )
                    
                    herb_qa_pairs.append({
                        "prompt": question,
                        "completion": answer
                    })
                    
                except KeyError as e:
                    print(f"Error formatting template for {local_name}: {e}")
                    continue
            
            # Add some additional variations
            additional_questions = [
                f"What is {local_name} used for?",
                f"How to use {local_name}?",
                f"Benefits of {local_name}",
                f"Dosage for {local_name}",
                f"Precautions for {local_name}"
            ]
            
            for q in additional_questions[:num_pairs_per_herb - len(herb_qa_pairs)]:
                answer = f"{local_name} ({herb.get('Scientific Name', 'Unknown')}) is used for: {herb.get('Uses', 'Unknown')}. Preparation: {herb.get('Preparation & Dosage', 'Unknown')}. Precautions: {herb.get('Side Effects / Precautions', 'Unknown')}."
                herb_qa_pairs.append({
                    "prompt": q,
                    "completion": answer
                })
            
            all_qa_pairs.extend(herb_qa_pairs[:num_pairs_per_herb])
        
        return all_qa_pairs
    
    def save_jsonl(self, qa_pairs: List[Dict[str, str]], output_path: str):
        """Save Q&A pairs in JSONL format"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for qa in qa_pairs:
                f.write(json.dumps(qa, ensure_ascii=False) + '\n')
    
    def save_train_test_split(self, qa_pairs: List[Dict[str, str]], 
                            train_ratio: float = 0.8, 
                            output_dir: str = "data"):
        """Split data into train and test sets"""
        random.shuffle(qa_pairs)
        split_idx = int(len(qa_pairs) * train_ratio)
        
        train_data = qa_pairs[:split_idx]
        test_data = qa_pairs[split_idx:]
        
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        # Save train and test sets
        self.save_jsonl(train_data, f"{output_dir}/train.jsonl")
        self.save_jsonl(test_data, f"{output_dir}/test.jsonl")
        
        print(f"Saved {len(train_data)} training examples to {output_dir}/train.jsonl")
        print(f"Saved {len(test_data)} test examples to {output_dir}/test.jsonl")
        
        return len(train_data), len(test_data)

def load_herbs_data(file_path: str) -> List[Dict[str, Any]]:
    """Load herbs data from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description="Generate Q&A pairs from herbs dataset")
    parser.add_argument("--input", default="herbs.json", help="Input herbs JSON file")
    parser.add_argument("--output-dir", default="data", help="Output directory for processed data")
    parser.add_argument("--qa-per-herb", type=int, default=10, help="Number of Q&A pairs per herb")
    parser.add_argument("--subset", type=int, help="Use only first N herbs for testing")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train/test split ratio")
    
    args = parser.parse_args()
    
    print("Loading herbs data...")
    herbs_data = load_herbs_data(args.input)
    
    if args.subset:
        herbs_data = herbs_data[:args.subset]
        print(f"Using subset of {len(herbs_data)} herbs")
    
    print(f"Loaded {len(herbs_data)} herbs")
    
    # Generate Q&A pairs
    print("Generating Q&A pairs...")
    generator = HerbalQAGenerator(herbs_data)
    qa_pairs = generator.generate_qa_pairs(args.qa_per_herb)
    
    print(f"Generated {len(qa_pairs)} Q&A pairs")
    
    # Save train/test split
    train_count, test_count = generator.save_train_test_split(
        qa_pairs, 
        args.train_ratio, 
        args.output_dir
    )
    
    # Save full dataset
    generator.save_jsonl(qa_pairs, f"{args.output_dir}/full_dataset.jsonl")
    
    print(f"\nDataset preparation complete!")
    print(f"Total Q&A pairs: {len(qa_pairs)}")
    print(f"Training examples: {train_count}")
    print(f"Test examples: {test_count}")
    print(f"Output directory: {args.output_dir}")

if __name__ == "__main__":
    main()
