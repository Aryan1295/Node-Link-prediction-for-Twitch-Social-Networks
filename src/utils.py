"""
Visualization and Analysis Utilities
Plots training curves, ROC curves, and metric comparisons.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List


# Style settings
sns.set_theme(style="whitegrid", font_scale=1.1)
COLORS = {"gcn": "#2196F3", "sage": "#4CAF50", "gat": "#FF5722"}


def plot_training_curves(all_results: List[Dict], save_dir: str):
    """
    Plot training loss and validation AUC curves for all models.

    Args:
        all_results: List of result dicts from train_model()
        save_dir: Directory to save plots
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for result in all_results:
        arch = result["arch"]
        color = COLORS.get(arch, "#666")
        history = result["history"]
        epochs = range(1, len(history["train_loss"]) + 1)

        axes[0].plot(epochs, history["train_loss"], label=arch.upper(),
                     color=color, linewidth=2)
        axes[1].plot(epochs, history["val_auc"], label=arch.upper(),
                     color=color, linewidth=2)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Training Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend()

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation AUC-ROC")
    axes[1].set_title("Validation AUC")
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved training curves to {path}")


def plot_roc_comparison(all_curves: Dict[str, Dict], save_dir: str):
    """
    Plot ROC curves for all models on the test set.

    Args:
        all_curves: Dict mapping arch name to curve data
        save_dir: Directory to save plots
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for arch, curves in all_curves.items():
        color = COLORS.get(arch, "#666")

        # ROC curve
        roc = curves["roc"]
        axes[0].plot(roc["fpr"], roc["tpr"], label=arch.upper(),
                     color=color, linewidth=2)

        # PR curve
        pr = curves["pr"]
        axes[1].plot(pr["recall"], pr["precision"], label=arch.upper(),
                     color=color, linewidth=2)

    # ROC
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve (Test Set)")
    axes[0].legend()

    # PR
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve (Test Set)")
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(save_dir, "roc_pr_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved ROC/PR curves to {path}")


def plot_metric_comparison(comparison_df: pd.DataFrame, save_dir: str):
    """
    Bar chart comparing key metrics across architectures.

    Args:
        comparison_df: DataFrame with architecture rows and metric columns
        save_dir: Directory to save plots
    """
    metrics = ["auc_roc", "avg_precision", "f1"]
    metric_labels = ["AUC-ROC", "Avg Precision", "F1 Score"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, metric, label in zip(axes, metrics, metric_labels):
        bars = ax.bar(
            comparison_df["architecture"],
            comparison_df[metric],
            color=[COLORS.get(a.lower(), "#666") for a in comparison_df["architecture"]],
            edgecolor="white",
            linewidth=1.5,
        )
        ax.set_title(label)
        ax.set_ylim(0, 1.05)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                    f"{height:.3f}", ha="center", va="bottom", fontsize=11)

    plt.tight_layout()
    path = os.path.join(save_dir, "metric_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved metric comparison to {path}")


def plot_embedding_tsne(model, data, device: str = "cpu", save_dir: str = "results"):
    """
    Visualize learned node embeddings using t-SNE.

    Args:
        model: Trained LinkPredictionModel
        data: Graph data with node features and labels
        device: Device string
        save_dir: Directory to save plot
    """
    from sklearn.manifold import TSNE
    import torch

    model.eval()
    data = data.to(device)

    with torch.no_grad():
        z = model.encode(data.x, data.edge_index).cpu().numpy()

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    z_2d = tsne.fit_transform(z)

    fig, ax = plt.subplots(figsize=(10, 8))

    if hasattr(data, "y") and data.y is not None:
        labels = data.y.cpu().numpy()
        scatter = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=labels, cmap="tab10",
                             s=10, alpha=0.7)
        plt.colorbar(scatter, ax=ax, label="Node Class")
    else:
        ax.scatter(z_2d[:, 0], z_2d[:, 1], s=10, alpha=0.7, color="#2196F3")

    ax.set_title("t-SNE of Learned Node Embeddings")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")

    path = os.path.join(save_dir, "embedding_tsne.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved t-SNE plot to {path}")


def save_results(all_results: List[Dict], comparison_df: pd.DataFrame, save_dir: str):
    """Save results to JSON and CSV."""
    # Save comparison table
    csv_path = os.path.join(save_dir, "comparison.csv")
    comparison_df.to_csv(csv_path, index=False)

    # Save detailed results (without non-serializable curve data)
    details = []
    for r in all_results:
        detail = {
            "arch": r["arch"],
            "test_metrics": r["test_metrics"],
            "best_epoch": r["best_epoch"],
            "train_time": r["train_time"],
            "total_params": r["total_params"],
        }
        details.append(detail)

    json_path = os.path.join(save_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(details, f, indent=2)

    print(f"Saved comparison CSV to {csv_path}")
    print(f"Saved detailed results to {json_path}")
