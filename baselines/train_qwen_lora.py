#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Baseline 3: Qwen3.5-4B LoRA Fine-tuning

Large Language Model baseline using LoRA (Low-Rank Adaptation) for parameter-
efficient fine-tuning. The model is trained as a generative classifier: given
a text, it generates "是" (yes) or "否" (no) to indicate lead generation content.

Usage:
    python train_qwen_lora.py [--versions v1 v2 ...] [--gpu 0]
    python train_qwen_lora.py --model_name /path/to/Qwen3.5-4B --versions v1 v2

Requirements:
    - transformers
    - peft
    - pandas, openpyxl
    - torch, datasets, sklearn
"""

import os
import argparse
import pandas as pd
import torch
import gc
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# ==============================================================================
# Configuration
# ==============================================================================

DEFAULT_VERSIONS = ["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8"]
DEFAULT_MODEL_NAME = "Qwen/Qwen3.5-4B"
DEFAULT_MAX_SEQ_LENGTH = 512
DEFAULT_BATCH_SIZE = 2
DEFAULT_GRAD_ACCUM = 4  # effective batch size = 2 * 4 = 8
DEFAULT_LR = 2e-4
DEFAULT_EPOCHS = 3
DEFAULT_WARMUP_RATIO = 0.1

# LoRA hyperparameters
DEFAULT_LORA_R = 16
DEFAULT_LORA_ALPHA = 32
DEFAULT_LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

TRAIN_FILE = "seen_set.xlsx"
TEST_FILE = "unseen_test_set.xlsx"

# Prompt template
PROMPT_TEMPLATE = """请判断以下文本是否为推广内容（lead generation）。

文本：{text}

请回答"是"或"否"。

答案："""


# ==============================================================================
# Data Loading
# ==============================================================================

def read_xlsx(filepath):
    """
    Read xlsx dataset file.

    Expected columns:
        - content: text content
        - is_leadgen: '是' (positive) or '否' (negative)

    Returns:
        DataFrame with columns ['text', 'labels']
    """
    df = pd.read_excel(filepath)
    result = pd.DataFrame()
    result["text"] = df["content"].astype(str)
    result["labels"] = df["is_leadgen"].map({"是": 1, "否": 0})
    result = result.dropna(subset=["text", "labels"]).reset_index(drop=True)
    result["labels"] = result["labels"].astype(int)
    return result


def format_instruction(text, label=None):
    """Format text as instruction prompt. If label is given, append the answer."""
    prompt = PROMPT_TEMPLATE.format(text=text)
    if label is not None:
        answer = "是" if label == 1 else "否"
        return prompt + answer
    return prompt


def preprocess_function(examples, tokenizer, max_seq_length):
    """Tokenize examples into model inputs with labels for causal LM training."""
    inputs = []
    for text, label in zip(examples["text"], examples["labels"]):
        formatted_text = format_instruction(text, label)
        inputs.append(formatted_text)

    model_inputs = tokenizer(
        inputs,
        max_length=max_seq_length,
        padding="max_length",
        truncation=True,
        return_tensors=None,
    )
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs


# ==============================================================================
# Model Loading
# ==============================================================================

def load_model_and_tokenizer(model_name, lora_r, lora_alpha, lora_dropout):
    """Load base model with LoRA adapter applied."""
    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
    )

    # Freeze base model
    for param in model.parameters():
        param.requires_grad = False

    # Apply LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, tokenizer


# ==============================================================================
# Evaluation
# ==============================================================================

def evaluate_model(model, tokenizer, eval_df, max_seq_length, device="cuda"):
    """
    Evaluate generative model by checking generated answers.

    The model generates a response after the prompt, and we check whether
    the output contains "是" or "否" to determine the predicted label.
    """
    model.eval()
    predictions = []
    labels = []

    for idx, row in eval_df.iterrows():
        text = row["text"]
        true_label = row["labels"]

        prompt = format_instruction(text)
        inputs = tokenizer(
            prompt, return_tensors="pt", max_length=max_seq_length, truncation=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Parse prediction from generated text
        if "答案：是" in generated_text or generated_text.strip().endswith("是"):
            pred_label = 1
        elif "答案：否" in generated_text or generated_text.strip().endswith("否"):
            pred_label = 0
        else:
            # Fallback: check last 20 characters
            tail = generated_text[-20:]
            if "是" in tail:
                pred_label = 1
            elif "否" in tail:
                pred_label = 0
            else:
                pred_label = -1  # unparseable

        predictions.append(pred_label)
        labels.append(true_label)

        if (idx + 1) % 100 == 0:
            print(f"  Evaluated {idx + 1}/{len(eval_df)} samples")

    # Filter valid predictions
    valid_indices = [i for i, p in enumerate(predictions) if p != -1]
    if not valid_indices:
        print("  No valid predictions!")
        return None

    valid_preds = [predictions[i] for i in valid_indices]
    valid_labels = [labels[i] for i in valid_indices]

    precision, recall, f1, _ = precision_recall_fscore_support(
        valid_labels, valid_preds, average="binary", zero_division=0
    )
    accuracy = accuracy_score(valid_labels, valid_preds)

    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "valid_count": len(valid_indices),
        "total_count": len(predictions),
    }
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  Valid: {len(valid_indices)}/{len(predictions)}")
    return results


# ==============================================================================
# Training
# ==============================================================================

def train_on_version(version, args):
    """Fine-tune Qwen with LoRA on a single dataset version."""
    print("\n" + "=" * 60)
    print(f"  Training Qwen-LoRA on dataset: {version}")
    print("=" * 60)

    data_dir = os.path.join(args.data_dir, version)
    output_dir = os.path.join(args.output_dir, f"qwen-lora-{version}")

    # Load data
    train_df = read_xlsx(os.path.join(data_dir, TRAIN_FILE))
    eval_df = read_xlsx(os.path.join(data_dir, TEST_FILE))

    print(f"Train: {len(train_df)} (pos: {(train_df['labels']==1).sum()}, neg: {(train_df['labels']==0).sum()})")
    print(f"Test:  {len(eval_df)} (pos: {(eval_df['labels']==1).sum()}, neg: {(eval_df['labels']==0).sum()})")

    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(eval_df)

    # Load model (fresh base model for each version)
    model, tokenizer = load_model_and_tokenizer(
        args.model_name, args.lora_r, args.lora_alpha, args.lora_dropout
    )

    # Preprocess
    train_dataset = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer, args.max_seq_length),
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    eval_dataset = eval_dataset.map(
        lambda x: preprocess_function(x, tokenizer, args.max_seq_length),
        batched=True,
        remove_columns=eval_dataset.column_names,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=10,
        save_total_limit=2,
        fp16=False,
        bf16=torch.cuda.is_bf16_supported(),
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        warmup_ratio=args.warmup_ratio,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
    )

    # Train
    trainer.train()

    # Save final model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to: {output_dir}")

    # Evaluate on test set
    print(f"\nEvaluating {version} on unseen test set...")
    eval_df_raw = read_xlsx(os.path.join(data_dir, TEST_FILE))
    results = evaluate_model(model, tokenizer, eval_df_raw, args.max_seq_length)

    # Save results
    if results:
        results_path = os.path.join(output_dir, "eval_results.txt")
        with open(results_path, "w") as f:
            f.write(f"Dataset Version: {version}\n")
            f.write(f"Method: Qwen-LoRA\n")
            f.write(f"Model: {args.model_name}\n")
            f.write(f"LoRA: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}\n")
            f.write(f"Train Size: {len(train_df)}\n")
            f.write(f"Test Size: {len(eval_df_raw)}\n\n")
            for k, v in results.items():
                f.write(f"{k}: {v}\n")
        print(f"Results saved to: {results_path}")

    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()

    return results


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Qwen-LoRA Baseline Training")
    parser.add_argument("--versions", nargs="+", default=DEFAULT_VERSIONS,
                        help="Dataset versions to train on")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME,
                        help="Qwen model name or path")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--grad_accum", type=int, default=DEFAULT_GRAD_ACCUM)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--max_seq_length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    parser.add_argument("--warmup_ratio", type=float, default=DEFAULT_WARMUP_RATIO)
    parser.add_argument("--lora_r", type=int, default=DEFAULT_LORA_R)
    parser.add_argument("--lora_alpha", type=int, default=DEFAULT_LORA_ALPHA)
    parser.add_argument("--lora_dropout", type=float, default=DEFAULT_LORA_DROPOUT)
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Base directory containing versioned datasets")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Base directory for saving models")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU device index to use")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    print("=" * 60)
    print("Qwen-LoRA Baseline Training")
    print(f"  Model: {args.model_name}")
    print(f"  Versions: {args.versions}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size} x {args.grad_accum} = {args.batch_size * args.grad_accum}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  LoRA: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    print("=" * 60)

    all_results = {}
    for version in args.versions:
        results = train_on_version(version, args)
        all_results[version] = results

    # Print summary
    print("\n" + "=" * 60)
    print("  Summary - Test Set (Unseen Tactics)")
    print("=" * 60)
    print(f"{'Version':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    print("-" * 50)
    for ver in args.versions:
        res = all_results.get(ver)
        if res:
            print(f"{ver:<10} {res['accuracy']:<10.4f} {res['precision']:<10.4f} {res['recall']:<10.4f} {res['f1']:<10.4f}")
        else:
            print(f"{ver:<10} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10}")

    import numpy as np
    valid_f1s = [all_results[v]["f1"] for v in args.versions if all_results.get(v)]
    if valid_f1s:
        print(f"\nAvg F1: {np.mean(valid_f1s):.4f} +/- {np.std(valid_f1s):.4f}")
    print("\nAll training completed!")


if __name__ == "__main__":
    main()
