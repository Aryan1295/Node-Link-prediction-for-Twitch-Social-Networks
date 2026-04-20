"""
Cross-Network Generalization Experiment
Train on one Twitch region, test on another to evaluate transfer.
"""

import os
import sys
import json
import torch
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from dataset import load_dataset, split_edges
from models import build_model
from train import train_model, get_device
from evaluate import evaluate_model


def run_cross_network(
    train_region: str = "DE",
    test_regions: list = None,
    arch: str = "gat",
    epochs: int = 200,
    save_dir: str = "results",
):
    """
    Train on one region, evaluate on others.

    Args:
        train_region: Region to train on
        test_regions: Regions to evaluate on
        arch: Architecture to use
        epochs: Training epochs
        save_dir: Output directory
    """
    if test_regions is None:
        test_regions = ["EN", "ES", "FR", "PT", "RU"]

    os.makedirs(save_dir, exist_ok=True)
    device = get_device()
    print(f"Using device: {device}")

    # Load and split training data
    print(f"\n{'='*60}")
    print(f"Training on Twitch{train_region}")
    print(f"{'='*60}")
    train_data_full = load_dataset(f"Twitch{train_region}")
    train_data, val_data, test_data = split_edges(train_data_full)

    # Train model
    result = train_model(
        arch=arch,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        in_channels=train_data_full.num_node_features,
        epochs=epochs,
        device=device,
        save_dir=save_dir,
    )

    # Evaluate on same-network test set
    rows = [{
        "train_region": train_region,
        "test_region": train_region,
        "auc_roc": result["test_metrics"]["auc_roc"],
        "avg_precision": result["test_metrics"]["avg_precision"],
        "f1": result["test_metrics"]["f1"],
        "transfer_type": "same-network",
    }]

    # Rebuild model for cross-network eval
    model = build_model(arch, train_data_full.num_node_features)
    model.load_state_dict(torch.load(result["model_path"], weights_only=True, map_location=device))
    model = model.to(device)

    # Evaluate on other regions
    for region in test_regions:
        if region == train_region:
            continue
        print(f"\n--- Testing on Twitch{region} ---")
        try:
            test_full = load_dataset(f"Twitch{region}")

            # Check feature dimension compatibility
            if test_full.num_node_features != train_data_full.num_node_features:
                print(f"  Skipping: feature dim mismatch "
                      f"({test_full.num_node_features} vs {train_data_full.num_node_features})")
                continue

            _, _, cross_test = split_edges(test_full)
            metrics = evaluate_model(model, cross_test, device)

            rows.append({
                "train_region": train_region,
                "test_region": region,
                "auc_roc": metrics["auc_roc"],
                "avg_precision": metrics["avg_precision"],
                "f1": metrics["f1"],
                "transfer_type": "cross-network",
            })
            print(f"  AUC: {metrics['auc_roc']:.4f} | AP: {metrics['avg_precision']:.4f}")
        except Exception as e:
            print(f"  Error: {e}")

    # Save results
    df = pd.DataFrame(rows)
    print(f"\n{'='*60}")
    print("CROSS-NETWORK RESULTS")
    print(f"{'='*60}")
    print(df.to_string(index=False))

    csv_path = os.path.join(save_dir, "cross_network_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved to {csv_path}")

    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-region", default="DE")
    parser.add_argument("--test-regions", nargs="+", default=["EN", "ES", "FR", "PT", "RU"])
    parser.add_argument("--arch", default="gat", choices=["gcn", "sage", "gat"])
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()

    run_cross_network(args.train_region, args.test_regions, args.arch, args.epochs)
