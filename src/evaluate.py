"""
Evaluation Metrics for Link Prediction
Computes AUC-ROC, Average Precision, Precision, Recall, and F1.
"""

import torch
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    precision_recall_curve,
)
from typing import Dict, Tuple
from torch_geometric.data import Data


@torch.no_grad()
def compute_predictions(model, data: Data, device: str = "cpu") -> Tuple[np.ndarray, np.ndarray]:
    """
    Run model on a data split and return predictions + labels.

    Args:
        model: Trained LinkPredictionModel
        data: PyG Data object with edge_label_index and edge_label
        device: Device to run on

    Returns:
        (predictions, labels) as numpy arrays
    """
    model.eval()
    data = data.to(device)

    logits = model(data.x, data.edge_index, data.edge_label_index)
    preds = torch.sigmoid(logits).cpu().numpy()
    labels = data.edge_label.cpu().numpy()

    return preds, labels


def compute_metrics(preds: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """
    Compute all link prediction evaluation metrics.

    Args:
        preds: Predicted probabilities
        labels: Ground truth binary labels
        threshold: Classification threshold for precision/recall/F1

    Returns:
        Dict with AUC, AP, precision, recall, F1
    """
    binary_preds = (preds >= threshold).astype(int)

    metrics = {
        "auc_roc": roc_auc_score(labels, preds),
        "avg_precision": average_precision_score(labels, preds),
        "precision": precision_score(labels, binary_preds, zero_division=0),
        "recall": recall_score(labels, binary_preds, zero_division=0),
        "f1": f1_score(labels, binary_preds, zero_division=0),
    }

    return {k: round(v, 4) for k, v in metrics.items()}


def get_curve_data(preds: np.ndarray, labels: np.ndarray) -> Dict:
    """Get ROC and PR curve data for plotting."""
    fpr, tpr, roc_thresholds = roc_curve(labels, preds)
    prec, rec, pr_thresholds = precision_recall_curve(labels, preds)

    return {
        "roc": {"fpr": fpr, "tpr": tpr, "thresholds": roc_thresholds},
        "pr": {"precision": prec, "recall": rec, "thresholds": pr_thresholds},
    }


@torch.no_grad()
def evaluate_model(model, data: Data, device: str = "cpu", threshold: float = 0.5) -> Dict[str, float]:
    """
    Full evaluation pipeline: run model and compute metrics.

    Args:
        model: Trained LinkPredictionModel
        data: PyG Data split (val or test)
        device: Device string
        threshold: Classification threshold

    Returns:
        Dict of evaluation metrics
    """
    preds, labels = compute_predictions(model, data, device)
    return compute_metrics(preds, labels, threshold)
