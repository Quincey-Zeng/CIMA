#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Baseline 2: BERT-VREx (Variance Risk Extrapolation)

BERT fine-tuning with VREx domain generalization objective. Within each dataset
version, positive samples are grouped into domains by their primary adversarial
tactic type. The VREx loss penalizes variance of per-domain losses to encourage
learning domain-invariant features.

Loss: L = mean(L_e) + lambda * Var(L_e)
where L_e is the average cross-entropy loss for domain e.

Lambda annealing: first `anneal_epochs` use lambda=0 (pure ERM warmup),
then linearly ramp to target lambda.

Usage:
    python train_bert_vrex.py [--versions v1 v2 ...] [--lambda_vrex 10.0] [--gpu 0]

Requirements:
    - transformers
    - pandas, openpyxl, numpy
    - torch, sklearn
"""

import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import gc
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# ==============================================================================
# Configuration
# ==============================================================================

DEFAULT_VERSIONS = ["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8"]
DEFAULT_EPOCHS = 14
DEFAULT_BATCH_SIZE = 8
DEFAULT_EVAL_BATCH_SIZE = 100
DEFAULT_LR = 4e-5
DEFAULT_MAX_SEQ_LENGTH = 128
DEFAULT_WARMUP_RATIO = 0.06
DEFAULT_LAMBDA = 10.0
DEFAULT_ANNEAL_EPOCHS = 2
MIN_DOMAIN_SAMPLES = 5

TRAIN_FILE = "seen_set.xlsx"
TEST_FILE = "unseen_test_set.xlsx"


# ==============================================================================
# Data Loading & Domain Assignment
# ==============================================================================

def read_xlsx(filepath):
    """
    Read xlsx dataset file.

    Expected columns:
        - content: text content
        - is_leadgen: '是' (positive) or '否' (negative)
        - 对抗手段: adversarial tactic types (comma-separated), '无' for negatives

    Returns:
        DataFrame with columns ['text', 'labels', 'tactic']
    """
    df = pd.read_excel(filepath)
    result = pd.DataFrame()
    result["text"] = df["content"].astype(str)
    result["labels"] = df["is_leadgen"].map({"是": 1, "否": 0})
    result["tactic"] = df["对抗手段"].fillna("无").astype(str)
    result = result.dropna(subset=["text", "labels"]).reset_index(drop=True)
    result["labels"] = result["labels"].astype(int)
    return result


def assign_domains(df, min_samples=MIN_DOMAIN_SAMPLES):
    """
    Assign domain IDs to each sample for VREx training.

    Strategy:
        - Positive samples: grouped by primary (first) adversarial tactic
        - Negative samples: uniformly distributed across all domains
        - Tactics with < min_samples positive examples merged into 'other'

    Args:
        df: DataFrame with columns ['text', 'labels', 'tactic']
        min_samples: minimum positive samples to form a standalone domain

    Returns:
        (df_with_domain_id, num_domains)
    """
    df = df.copy()
    pos_mask = df["labels"] == 1

    # Extract primary tactic (first one if comma-separated)
    df["primary_tactic"] = df["tactic"].apply(
        lambda x: x.split(",")[0].strip() if x != "无" else "无"
    )

    # Count positive samples per tactic
    pos_tactic_counts = df.loc[pos_mask, "primary_tactic"].value_counts()
    small_tactics = set(
        pos_tactic_counts[pos_tactic_counts < min_samples].index
    )
    small_tactics.add("无")

    # Valid domains (sufficient samples)
    valid_tactics = sorted([
        t for t in pos_tactic_counts.index if t not in small_tactics
    ])

    if not valid_tactics:
        valid_tactics = ["all"]

    # Assign domain IDs to positive samples
    tactic_to_id = {t: i for i, t in enumerate(valid_tactics)}
    other_id = len(valid_tactics)  # 'other' domain

    domain_ids = []
    for _, row in df.iterrows():
        if row["labels"] == 1:
            pt = row["primary_tactic"]
            domain_ids.append(tactic_to_id.get(pt, other_id))
        else:
            domain_ids.append(-1)  # placeholder for negatives

    df["domain_id"] = domain_ids

    # Distribute negative samples uniformly across domains
    num_domains = other_id + 1
    neg_indices = df[df["domain_id"] == -1].index.tolist()
    np.random.seed(42)
    np.random.shuffle(neg_indices)
    for i, idx in enumerate(neg_indices):
        df.loc[idx, "domain_id"] = i % num_domains

    df["domain_id"] = df["domain_id"].astype(int)

    # Print domain statistics
    domain_names = valid_tactics + ["other"]
    print(f"  Domains ({len(domain_names)}): {domain_names}")
    for did, dname in enumerate(domain_names):
        d_df = df[df["domain_id"] == did]
        pos_count = (d_df["labels"] == 1).sum()
        neg_count = (d_df["labels"] == 0).sum()
        print(f"    [{did}] {dname}: {len(d_df)} samples (pos={pos_count}, neg={neg_count})")

    return df, num_domains


# ==============================================================================
# Dataset
# ==============================================================================

class TextClassificationDataset(TorchDataset):
    """PyTorch Dataset for BERT classification with domain IDs."""

    def __init__(self, texts, labels, domain_ids, tokenizer, max_length):
        self.encodings = tokenizer(
            texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors=None,
        )
        self.labels = labels
        self.domain_ids = domain_ids

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        item["domain_ids"] = torch.tensor(self.domain_ids[idx], dtype=torch.long)
        return item


# ==============================================================================
# VREx Trainer
# ==============================================================================

class VRExTrainer(Trainer):
    """
    Custom Trainer implementing VREx (Variance Risk Extrapolation).

    Loss = mean(domain_losses) + lambda * var(domain_losses)

    With lambda annealing: first `anneal_epochs` use lambda=0 (ERM warmup),
    then linearly increase to target lambda.
    """

    def __init__(self, *args, lambda_vrex=1.0, num_domains=1, anneal_epochs=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_vrex = lambda_vrex
        self.num_domains = num_domains
        self.anneal_epochs = anneal_epochs
        self.total_epochs = self.args.num_train_epochs

    def _get_current_lambda(self):
        """Compute current lambda based on training progress."""
        if self.state.epoch is None:
            return 0.0
        current_epoch = self.state.epoch
        if current_epoch <= self.anneal_epochs:
            return 0.0
        ramp = (current_epoch - self.anneal_epochs) / max(
            self.total_epochs - self.anneal_epochs, 1
        )
        return self.lambda_vrex * min(ramp, 1.0)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        domain_ids = inputs.pop("domain_ids")
        labels = inputs["labels"]

        outputs = model(**inputs)
        logits = outputs.logits

        # Per-sample cross-entropy loss
        per_sample_loss = F.cross_entropy(logits, labels, reduction="none")

        # Group by domain
        domain_losses = []
        for d in range(self.num_domains):
            mask = domain_ids == d
            if mask.sum() > 0:
                domain_losses.append(per_sample_loss[mask].mean())

        current_lambda = self._get_current_lambda()

        if len(domain_losses) < 2 or current_lambda == 0.0:
            # Fewer than 2 domains in batch or annealing phase: fallback to ERM
            loss = per_sample_loss.mean()
        else:
            domain_losses = torch.stack(domain_losses)
            mean_loss = domain_losses.mean()
            var_loss = domain_losses.var()
            loss = mean_loss + current_lambda * var_loss

        return (loss, outputs) if return_outputs else loss


# ==============================================================================
# Evaluation
# ==============================================================================

def evaluate_model(model, tokenizer, eval_df, max_seq_length, batch_size=100, device="cuda"):
    """Evaluate model on a dataset and return metrics."""
    model.eval()
    predictions = []
    labels_list = []

    texts = eval_df["text"].tolist()
    labels = eval_df["labels"].tolist()

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_labels = labels[i:i + batch_size]

        inputs = tokenizer(
            batch_texts,
            max_length=max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1).cpu().tolist()

        predictions.extend(preds)
        labels_list.extend(batch_labels)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_list, predictions, average="binary", zero_division=0
    )
    accuracy = accuracy_score(labels_list, predictions)

    tp = sum(1 for p, l in zip(predictions, labels_list) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(predictions, labels_list) if p == 1 and l == 0)
    tn = sum(1 for p, l in zip(predictions, labels_list) if p == 0 and l == 0)
    fn = sum(1 for p, l in zip(predictions, labels_list) if p == 0 and l == 1)

    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "F1_score": f1,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }

    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")

    return results


# ==============================================================================
# Training
# ==============================================================================

def train_on_version(version, lambda_vrex, args):
    """Train BERT-VREx on a single dataset version."""
    print("\n" + "=" * 60)
    print(f"  Training BERT-VREx on dataset: {version} (lambda={lambda_vrex})")
    print("=" * 60)

    data_dir = os.path.join(args.data_dir, version)
    output_dir = os.path.join(args.output_dir, f"bert-vrex-{version}-lambda{lambda_vrex}")

    # Load data
    train_df = read_xlsx(os.path.join(data_dir, TRAIN_FILE))
    eval_df = read_xlsx(os.path.join(data_dir, TEST_FILE))

    print(f"Train: {len(train_df)} (pos: {(train_df['labels']==1).sum()}, neg: {(train_df['labels']==0).sum()})")
    print(f"Test:  {len(eval_df)} (pos: {(eval_df['labels']==1).sum()}, neg: {(eval_df['labels']==0).sum()})")

    # Assign domains
    print("\nAssigning domains...")
    train_df, num_domains = assign_domains(train_df, min_samples=args.min_domain_samples)

    # Load model and tokenizer
    print("\nLoading model...")
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model)
    model = BertForSequenceClassification.from_pretrained(
        args.pretrained_model, num_labels=2
    )

    # Create dataset
    print("Tokenizing...")
    train_dataset = TextClassificationDataset(
        texts=train_df["text"].tolist(),
        labels=train_df["labels"].tolist(),
        domain_ids=train_df["domain_id"].tolist(),
        tokenizer=tokenizer,
        max_length=args.max_seq_length,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.lr,
        weight_decay=0.0,
        warmup_ratio=args.warmup_ratio,
        logging_steps=50,
        save_strategy="no",
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    # Create VREx Trainer
    trainer = VRExTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        lambda_vrex=lambda_vrex,
        num_domains=num_domains,
        anneal_epochs=args.anneal_epochs,
    )

    # Train
    print(f"\nStarting training (lambda={lambda_vrex}, anneal_epochs={args.anneal_epochs})...")
    trainer.train()

    # Save model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to: {output_dir}")

    # Evaluate
    print(f"\n=== {version} - Train Set Results ===")
    train_results = evaluate_model(
        model, tokenizer, train_df, args.max_seq_length,
        batch_size=args.eval_batch_size
    )

    print(f"\n=== {version} - Test Set Results ===")
    test_results = evaluate_model(
        model, tokenizer, eval_df, args.max_seq_length,
        batch_size=args.eval_batch_size
    )

    # Save results
    results_path = os.path.join(output_dir, "eval_results.txt")
    with open(results_path, "w") as f:
        f.write(f"Dataset Version: {version}\n")
        f.write(f"Method: BERT-VREx (lambda={lambda_vrex}, anneal_epochs={args.anneal_epochs})\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Train Size: {len(train_df)}\n")
        f.write(f"Test Size: {len(eval_df)}\n")
        f.write(f"Num Domains: {num_domains}\n\n")
        f.write("=== Train Set ===\n")
        for k, v in train_results.items():
            f.write(f"{k}: {v}\n")
        f.write("\n=== Test Set ===\n")
        for k, v in test_results.items():
            f.write(f"{k}: {v}\n")
    print(f"Results saved to: {results_path}")

    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()

    return train_results, test_results


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="BERT-VREx Baseline Training")
    parser.add_argument("--versions", nargs="+", default=DEFAULT_VERSIONS,
                        help="Dataset versions to train on")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--eval_batch_size", type=int, default=DEFAULT_EVAL_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--max_seq_length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    parser.add_argument("--warmup_ratio", type=float, default=DEFAULT_WARMUP_RATIO)
    parser.add_argument("--lambda_vrex", type=float, default=DEFAULT_LAMBDA,
                        help="VREx penalty strength")
    parser.add_argument("--anneal_epochs", type=int, default=DEFAULT_ANNEAL_EPOCHS,
                        help="Number of warmup epochs with lambda=0")
    parser.add_argument("--min_domain_samples", type=int, default=MIN_DOMAIN_SAMPLES,
                        help="Minimum positive samples to form a domain")
    parser.add_argument("--pretrained_model", type=str,
                        default="bert-base-multilingual-uncased",
                        help="Pretrained model name or path")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Base directory containing versioned datasets")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Base directory for saving models")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU device index to use")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    print("=" * 60)
    print("BERT-VREx Baseline Training")
    print(f"  Versions: {args.versions}")
    print(f"  Lambda: {args.lambda_vrex}")
    print(f"  Anneal Epochs: {args.anneal_epochs}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.lr}")
    print("=" * 60)

    all_test_results = {}

    for version in args.versions:
        _, test_res = train_on_version(version, args.lambda_vrex, args)
        all_test_results[version] = test_res

    # Print summary
    print("\n" + "=" * 60)
    print(f"  Summary - Lambda={args.lambda_vrex} - Test Set (Unseen Tactics)")
    print("=" * 60)
    print(f"{'Version':<8} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    print("-" * 48)
    for ver in args.versions:
        r = all_test_results[ver]
        print(f"{ver:<8} {r['accuracy']:<10.4f} {r['precision']:<10.4f} {r['recall']:<10.4f} {r['F1_score']:<10.4f}")

    f1_scores = [all_test_results[v]["F1_score"] for v in args.versions]
    print(f"\nAvg F1: {np.mean(f1_scores):.4f} +/- {np.std(f1_scores):.4f}")
    print("\nAll training completed!")


if __name__ == "__main__":
    main()
