"""
Training Loop for Link Prediction Models
Supports training multiple architectures and comparing results.
"""

import argparse
import json
import os
import time
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Optional

from dataset import load_dataset, split_edges
from models import build_model
from evaluate import evaluate_model, compute_predictions, compute_metrics, get_curve_data
from utils import plot_training_curves, plot_roc_comparison, plot_metric_comparison, save_results


def get_device() -> str:
    """Detect best available device (MPS for Mac, CUDA, or CPU)."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train_epoch(model, train_data, optimizer, device: str) -> float:
    """
    Train for one epoch.

    Args:
        model: LinkPredictionModel
        train_data: Training split with edge_label_index and edge_label
        optimizer: Torch optimizer
        device: Device string

    Returns:
        Training loss for the epoch
    """
    model.train()
    train_data = train_data.to(device)

    optimizer.zero_grad()

    # Forward pass: predict on labeled edges
    logits = model(train_data.x, train_data.edge_index, train_data.edge_label_index)
    labels = train_data.edge_label.float()

    # Binary cross-entropy loss
    loss = F.binary_cross_entropy_with_logits(logits, labels)

    loss.backward()
    optimizer.step()

    return loss.item()


def train_model(
    arch: str,
    train_data,
    val_data,
    test_data,
    in_channels: int,
    hidden_channels: int = 128,
    out_channels: int = 64,
    num_layers: int = 2,
    heads: int = 4,
    lr: float = 0.005,
    weight_decay: float = 1e-4,
    epochs: int = 200,
    patience: int = 20,
    device: str = "cpu",
    save_dir: str = "results",
) -> Dict:
    """
    Full training pipeline for a single model.

    Includes early stopping based on validation AUC.

    Args:
        arch: Model architecture (gcn, sage, gat)
        train_data, val_data, test_data: Edge-split data
        in_channels: Input feature dimension
        Other args: Hyperparameters

    Returns:
        Dict with training history, test metrics, and model path
    """
    print(f"\n{'='*60}")
    print(f"Training {arch.upper()}")
    print(f"{'='*60}")

    model = build_model(arch, in_channels, hidden_channels, out_channels,
                        num_layers, heads)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Training history
    history = {
        "train_loss": [],
        "val_auc": [],
        "val_ap": [],
    }

    best_val_auc = 0
    best_epoch = 0
    patience_counter = 0
    model_path = os.path.join(save_dir, f"{arch}_best.pt")

    start_time = time.time()

    for epoch in tqdm(range(1, epochs + 1), desc=f"{arch.upper()} training"):
        # Train
        loss = train_epoch(model, train_data, optimizer, device)
        history["train_loss"].append(loss)

        # Validate
        val_metrics = evaluate_model(model, val_data, device)
        history["val_auc"].append(val_metrics["auc_roc"])
        history["val_ap"].append(val_metrics["avg_precision"])

        # Early stopping check
        if val_metrics["auc_roc"] > best_val_auc:
            best_val_auc = val_metrics["auc_roc"]
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1

        # Print progress every 25 epochs
        if epoch % 25 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | Loss: {loss:.4f} | "
                  f"Val AUC: {val_metrics['auc_roc']:.4f} | "
                  f"Val AP: {val_metrics['avg_precision']:.4f}")

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch} (best: {best_epoch})")
            break

    train_time = time.time() - start_time

    # Load best model and evaluate on test set
    model.load_state_dict(torch.load(model_path, weights_only=True))
    test_metrics = evaluate_model(model, test_data, device)

    print(f"\n  Training time: {train_time:.1f}s")
    print(f"  Best val AUC at epoch {best_epoch}: {best_val_auc:.4f}")
    print(f"  Test results:")
    for k, v in test_metrics.items():
        print(f"    {k}: {v:.4f}")

    # Get curve data for plots
    test_preds, test_labels = compute_predictions(model, test_data, device)
    curves = get_curve_data(test_preds, test_labels)

    return {
        "arch": arch,
        "history": history,
        "test_metrics": test_metrics,
        "best_epoch": best_epoch,
        "train_time": train_time,
        "model_path": model_path,
        "curves": curves,
        "total_params": sum(p.numel() for p in model.parameters()),
    }


def run_comparison(
    dataset_name: str = "Cora",
    architectures: List[str] = None,
    epochs: int = 200,
    hidden_channels: int = 128,
    out_channels: int = 64,
    lr: float = 0.005,
    save_dir: str = "results",
) -> pd.DataFrame:
    """
    Train and compare multiple GNN architectures.

    Args:
        dataset_name: Name of the dataset to use
        architectures: List of architectures to compare
        Other args: Shared hyperparameters

    Returns:
        Comparison DataFrame with test metrics for each model
    """
    if architectures is None:
        architectures = ["gcn", "sage", "gat"]

    os.makedirs(save_dir, exist_ok=True)
    device = get_device()
    print(f"Using device: {device}")

    # Load data
    data = load_dataset(dataset_name)
    train_data, val_data, test_data = split_edges(data)

    # Train each architecture
    all_results = []
    all_curves = {}

    for arch in architectures:
        result = train_model(
            arch=arch,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            in_channels=data.num_node_features,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            lr=lr,
            epochs=epochs,
            device=device,
            save_dir=save_dir,
        )
        all_results.append(result)
        all_curves[arch] = result["curves"]

    # Build comparison table
    rows = []
    for r in all_results:
        row = {"architecture": r["arch"].upper(), "params": r["total_params"],
               "best_epoch": r["best_epoch"], "train_time_s": round(r["train_time"], 1)}
        row.update(r["test_metrics"])
        rows.append(row)

    comparison_df = pd.DataFrame(rows)
    print(f"\n{'='*60}")
    print("COMPARISON RESULTS")
    print(f"{'='*60}")
    print(comparison_df.to_string(index=False))

    # Save results
    save_results(all_results, comparison_df, save_dir)

    # Generate plots
    plot_training_curves(all_results, save_dir)
    plot_roc_comparison(all_curves, save_dir)
    plot_metric_comparison(comparison_df, save_dir)

    return comparison_df


def main():
    parser = argparse.ArgumentParser(description="Train GNN Link Prediction Models")
    parser.add_argument("--dataset", type=str, default="TwitchEN",
                        choices=["Cora", "CiteSeer", "TwitchDE", "TwitchEN", "TwitchES", "TwitchFR", "TwitchPT", "TwitchRU"])
    parser.add_argument("--models", nargs="+", default=["gcn", "sage", "gat"], choices=["gcn", "sage", "gat"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--save-dir", type=str, default="results")
    args = parser.parse_args()

    run_comparison(
        dataset_name=args.dataset,
        architectures=args.models,
        epochs=args.epochs,
        hidden_channels=args.hidden,
        out_channels=args.embed_dim,
        lr=args.lr,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
