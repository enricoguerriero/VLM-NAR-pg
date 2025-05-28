#!/usr/bin/env python
import json
import os
from argparse import ArgumentParser

import torch
from dotenv import load_dotenv
from huggingface_hub import login

from src.utils import (
    setup_logging,
    set_global_seed,
    load_config,
    compute_metrics,
    setup_wandb,
    log_test_wandb
)

def main():
    parser = ArgumentParser(
        description="Read judged predictions, compute metrics, and log to W&B"
    )
    parser.add_argument(
        "--model_name", type=str, required=True,
        help="Name of the model that produced the captions"
    )
    parser.add_argument(
        "--predictions_file", type=str, default="predictions.ndjson",
        help="Path to the NDJSON file with judge outputs"
    )
    args = parser.parse_args()

    # ——— Logging & W&B init ———
    logger = setup_logging(args.model_name, "binary_zero_shot")
    logger.info("→ Loading predictions and computing metrics")
    
    load_dotenv()
    login(os.getenv("HUGGINGFACE_TOKEN"))
    set_global_seed()

    config = load_config(args.model_name)
    config["test_type"] = "0-shot-per-label"
    wandb_run = setup_wandb(args.model_name, config)

    # ——— Read predictions ———
    entries = []
    with open(args.predictions_file, "r", encoding="utf-8") as f:
        for line in f:
            entries.append(json.loads(line))

    # Basic dataset statistics
    total_entries = len(entries)
    unique_clip_indices = len({e["clip_idx"] for e in entries})

    max_index = max(entry["clip_idx"] for entry in entries) + 1
    num_classes = 4
    logits = torch.zeros(max_index, num_classes)
    labels = torch.zeros(max_index, num_classes)

    for entry in entries:
        i = entry["clip_idx"]
        c = entry["class_idx"]
        if i >= logits.shape[0] or c >= num_classes:
            logger.warning(f"Invalid index i={i}, c={c}")
            continue
        logits[i, c] = float(entry["pred"])
        labels[i, c] = float(entry["label"])

    # Dataset description logging -------------------------------------------
    total_true_labels = int(labels.sum().item())
    true_labels_per_class = labels.sum(dim=0).tolist()

    logger.info(
        f"Dataset summary: {total_entries} total judged entries across "
        f"{unique_clip_indices} unique clips and {num_classes} classes."
    )
    logger.info(f"Total positive labels (all classes combined): {total_true_labels}")
    for class_idx, count in enumerate(true_labels_per_class):
        logger.info(f"  • Class {class_idx}: {int(count)} true labels")

    # ——— Compute & log metrics ———
    metrics = compute_metrics(logits=logits, labels=labels, threshold=0.5)
    logger.info(f"Metrics computed successfully. F1 macro: {metrics['f1_macro']:.4f}")

    # Log metrics + data summary to W&B
    log_test_wandb(wandb_run, metrics, 0)
    wandb_run.finish()
    logger.info("✅ Metrics and data summary logged to W&B. All done!")

if __name__ == "__main__":
    main()
