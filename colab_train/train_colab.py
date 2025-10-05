#!/usr/bin/env python3
"""
Minimal training script for Colab.
Uses LoRA and supports CPU/GPU auto selection.
"""

import argparse
import json
from pathlib import Path
import logging
import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType
from torch.nn.utils.rnn import pad_sequence


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("colab_train")


class CustomDataCollator:
    def __init__(self, tokenizer, pad_token_id: int):
        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        input_ids = [torch.tensor(f["input_ids"]) for f in features]
        labels = [torch.tensor(f["labels"]) for f in features]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        attention_mask = (input_ids != self.pad_token_id).long()
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def load_tokenizer_and_model(model_name: str, device: str, use_lora: bool):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )

    if use_lora:
        target_modules = ["c_attn"] if "distilgpt2" in model_name else ["q_proj", "v_proj"]
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_cfg)

    return tokenizer, model


def prepare_dataset(tokenizer, data_path: str, max_length: int) -> Dataset:
    # Detect jsonl with fields prompt/completion
    ds = load_dataset("json", data_files=data_path, split="train")

    def tokenize_function(batch):
        if "text" in batch:
            texts = batch["text"]
        elif "prompt" in batch and "completion" in batch:
            texts = [f"Human: {p}\nAssistant: {c}<|endoftext|>" for p, c in zip(batch["prompt"], batch["completion"])]
        else:
            # Fallback: join all fields
            texts = [json.dumps(row, ensure_ascii=False) for row in batch]

        enc = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors=None,
        )
        enc["labels"] = enc["input_ids"].copy()
        return enc

    ds_tok = ds.map(tokenize_function, batched=True, remove_columns=ds.column_names)
    return ds_tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilgpt2")
    ap.add_argument("--train-data", default="data/train.jsonl")
    ap.add_argument("--eval-data", default="data/test.jsonl")
    ap.add_argument("--output-dir", default="model")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--learning-rate", type=float, default=5e-5)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--use-lora", action="store_true")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"]) 
    ap.add_argument("--subset", type=int)
    args = ap.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    tokenizer, model = load_tokenizer_and_model(args.model, device, args.use_lora)

    train_ds = prepare_dataset(tokenizer, args.train_data, args.max_length)
    eval_ds = None
    if Path(args.eval_data).exists():
        eval_ds = prepare_dataset(tokenizer, args.eval_data, args.max_length)

    if args.subset:
        train_ds = train_ds.select(range(min(args.subset, len(train_ds))))
        if eval_ds:
            eval_ds = eval_ds.select(range(min(args.subset // 4, len(eval_ds))))

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=100,
        logging_steps=20,
        eval_steps=200 if eval_ds else None,
        save_steps=200,
        evaluation_strategy="steps" if eval_ds else "no",
        save_strategy="steps",
        load_best_model_at_end=True if eval_ds else False,
        fp16=(device == "cuda"),
        dataloader_pin_memory=(device == "cuda"),
        remove_unused_columns=False,
    )

    collator = CustomDataCollator(tokenizer, tokenizer.pad_token_id)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )

    logger.info("Starting training...")
    trainer.train()
    logger.info("Saving model...")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    with open(Path(args.output_dir) / "training_info.json", "w") as f:
        json.dump({
            "model": args.model,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "max_length": args.max_length,
            "use_lora": args.use_lora,
            "device": device,
            "train_samples": len(train_ds),
            "eval_samples": len(eval_ds) if eval_ds else 0,
        }, f, indent=2)

    logger.info("Done.")


if __name__ == "__main__":
    main()


