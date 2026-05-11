#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Baseline 1: BERT-ERM (Empirical Risk Minimization)

Standard BERT fine-tuning for binary classification (lead generation detection).
Trains independently on each dataset version and evaluates on the corresponding
unseen test set to measure out-of-distribution generalization.

Usage:
    python train_bert_erm.py [--versions v1 v2 ...] [--epochs 14] [--gpu 0]

Requirements:
    - simpletransformers
    - pandas, openpyxl
    - torch
"""

import os
import argparse
import pandas as pd
import torch
import gc
from simpletransformers.classification import ClassificationModel

# ==============================================================================
# Configuration
# ==============================================================================

DEFAULT_VERSIONS = ["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8"]
DEFAULT_EPOCHS = 14
DEFAULT_BATCH_SIZE = 8
DEFAULT_LR = 4e-5
DEFAULT_MAX_SEQ_LENGTH = 128

TRAIN_FILE = "seen_set.xlsx"
TEST_FILE = "unseen_test_set.xlsx"


# ==============================================================================
# Data Loading
# ==============================================================================

def read_xlsx(filepath):
    """
    Read xlsx dataset file and return DataFrame in simpletransformers format.

    Expected columns in xlsx:
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


# ==============================================================================
# Metrics
# ==============================================================================

def compute_metrics(result):
    """Compute precision, recall, F1 from tp/fp/tn/fn."""
    result = dict(result)
    tp, fp, tn, fn = result["tp"], result["fp"], result["tn"], result["fn"]
    result["accuracy"] = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    result["precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0
    result["recall"] = tp / (tp + fn) if (tp + fn) > 0 else 0
    if result["precision"] + result["recall"] > 0:
        result["F1_score"] = (
            2 * result["precision"] * result["recall"]
            / (result["precision"] + result["recall"])
        )
    else:
        result["F1_score"] = 0
    return result


# ==============================================================================
# Training
# ==============================================================================

def train_on_version(version, args):
    """Fine-tune BERT on a single dataset version and evaluate."""
    print("\n" + "=" * 60)
    print(f"  Training BERT-ERM on dataset: {version}")
    print("=" * 60)

    data_dir = os.path.join(args.data_dir, version)
    output_dir = os.path.join(args.output_dir, f"bert-erm-{version}")

    # Load data
    train_df = read_xlsx(os.path.join(data_dir, TRAIN_FILE))
    eval_df = read_xlsx(os.path.join(data_dir, TEST_FILE))

    print(f"Train: {len(train_df)} (pos: {(train_df['labels']==1).sum()}, neg: {(train_df['labels']==0).sum()})")
    print(f"Test:  {len(eval_df)} (pos: {(eval_df['labels']==1).sum()}, neg: {(eval_df['labels']==0).sum()})")

    # Training configuration
    train_args = {
        "num_train_epochs": args.epochs,
        "train_batch_size": args.batch_size,
        "learning_rate": args.lr,
        "max_seq_length": args.max_seq_length,
        "output_dir": output_dir,
        "overwrite_output_dir": True,
        "save_steps": -1,
        "save_model_every_epoch": False,
        "use_multiprocessing": False,
        "use_multiprocessing_for_evaluation": False,
        "dataloader_num_workers": 0,
    }

    # Train
    model = ClassificationModel(
        "bert", args.pretrained_model, args=train_args
    )
    model.train_model(train_df)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Evaluate with the trained model
    eval_args = {
        "use_multiprocessing": False,
        "use_multiprocessing_for_evaluation": False,
        "dataloader_num_workers": 0,
    }
    model = ClassificationModel("bert", output_dir, args=eval_args)

    # Evaluate on train set
    print(f"\n=== {version} - Train Set Results ===")
    train_result, _, _ = model.eval_model(train_df)
    train_result = compute_metrics(train_result)
    for k in ["accuracy", "precision", "recall", "F1_score"]:
        print(f"  {k}: {train_result[k]:.4f}")

    # Evaluate on test set (unseen tactics)
    print(f"\n=== {version} - Test Set Results ===")
    test_result, _, _ = model.eval_model(eval_df)
    test_result = compute_metrics(test_result)
    for k in ["accuracy", "precision", "recall", "F1_score"]:
        print(f"  {k}: {test_result[k]:.4f}")

    # Save results
    results_path = os.path.join(output_dir, "eval_results.txt")
    with open(results_path, "w") as f:
        f.write(f"Dataset Version: {version}\n")
        f.write(f"Method: BERT-ERM\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Train Size: {len(train_df)}\n")
        f.write(f"Test Size: {len(eval_df)}\n\n")
        f.write("=== Train Set ===\n")
        for k, v in train_result.items():
            f.write(f"{k}: {v}\n")
        f.write("\n=== Test Set ===\n")
        for k, v in test_result.items():
            f.write(f"{k}: {v}\n")
    print(f"Results saved to: {results_path}")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return train_result, test_result


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="BERT-ERM Baseline Training")
    parser.add_argument("--versions", nargs="+", default=DEFAULT_VERSIONS,
                        help="Dataset versions to train on")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--max_seq_length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
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
    print("BERT-ERM Baseline Training")
    print(f"  Versions: {args.versions}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Max Seq Length: {args.max_seq_length}")
    print(f"  Pretrained Model: {args.pretrained_model}")
    print("=" * 60)

    all_test_results = {}

    for version in args.versions:
        _, test_res = train_on_version(version, args)
        all_test_results[version] = test_res

    # Print summary
    print("\n" + "=" * 60)
    print("  Summary - Test Set (Unseen Tactics)")
    print("=" * 60)
    print(f"{'Version':<8} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    print("-" * 48)
    for ver in args.versions:
        r = all_test_results[ver]
        print(f"{ver:<8} {r['accuracy']:<10.4f} {r['precision']:<10.4f} {r['recall']:<10.4f} {r['F1_score']:<10.4f}")

    # Average
    f1_scores = [all_test_results[v]["F1_score"] for v in args.versions]
    import numpy as np
    print(f"\nAvg F1: {np.mean(f1_scores):.4f} +/- {np.std(f1_scores):.4f}")
    print("\nAll training completed!")


if __name__ == "__main__":
    main()
