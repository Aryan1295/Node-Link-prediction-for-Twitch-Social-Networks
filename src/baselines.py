"""
Heuristic Baselines for Link Prediction
Non-GNN methods to provide context for how much value the GNN adds.
Implements Common Neighbors, Jaccard Coefficient, and Adamic-Adar Index.
"""

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_scipy_sparse_matrix
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from typing import Dict, List, Tuple
import scipy.sparse as sp


def _build_adj(data: Data) -> sp.csr_matrix:
    """Build scipy sparse adjacency from training edge_index."""
    # return to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes)
    return to_scipy_sparse_matrix(data.edge_index.cpu(), num_nodes=data.num_nodes).tocsr()


def common_neighbors_score(adj: sp.csr_matrix, src: int, dst: int) -> float:
    """Count shared neighbors between src and dst."""
    n_src = set(adj[src].indices)
    n_dst = set(adj[dst].indices)
    return len(n_src & n_dst)


def jaccard_score_pair(adj: sp.csr_matrix, src: int, dst: int) -> float:
    """Jaccard coefficient: |intersection| / |union| of neighbor sets."""
    n_src = set(adj[src].indices)
    n_dst = set(adj[dst].indices)
    union = n_src | n_dst
    if len(union) == 0:
        return 0.0
    return len(n_src & n_dst) / len(union)


def adamic_adar_score(adj: sp.csr_matrix, src: int, dst: int) -> float:
    """Adamic-Adar: sum of 1/log(degree) for common neighbors."""
    n_src = set(adj[src].indices)
    n_dst = set(adj[dst].indices)
    common = n_src & n_dst
    score = 0.0
    for w in common:
        deg = adj[w].nnz
        if deg > 1:
            score += 1.0 / np.log(deg)
    return score


def evaluate_baseline(
    train_data: Data,
    test_data: Data,
    method: str = "common_neighbors",
) -> Dict[str, float]:
    """
    Evaluate a heuristic baseline on the test set.

    Args:
        train_data: Training split (used to build adjacency)
        test_data: Test split with edge_label_index and edge_label
        method: One of 'common_neighbors', 'jaccard', 'adamic_adar'

    Returns:
        Dict with auc_roc, avg_precision, f1
    """
    adj = _build_adj(train_data)

    score_fn = {
        "common_neighbors": common_neighbors_score,
        "jaccard": jaccard_score_pair,
        "adamic_adar": adamic_adar_score,
    }[method]

    # edge_label_index = test_data.edge_label_index.numpy()
    # labels = test_data.edge_label.numpy()
    edge_label_index = test_data.edge_label_index.cpu().numpy()
    labels = test_data.edge_label.cpu().numpy()

    scores = []
    for i in range(edge_label_index.shape[1]):
        src, dst = edge_label_index[0, i], edge_label_index[1, i]
        scores.append(score_fn(adj, src, dst))

    scores = np.array(scores)

    # Normalize to [0, 1] for threshold-based metrics
    s_max = scores.max()
    if s_max > 0:
        scores_norm = scores / s_max
    else:
        scores_norm = scores

    binary_preds = (scores_norm >= 0.5).astype(int)

    metrics = {
        "auc_roc": roc_auc_score(labels, scores) if scores.std() > 0 else 0.5,
        "avg_precision": average_precision_score(labels, scores) if scores.std() > 0 else 0.5,
        "f1": f1_score(labels, binary_preds, zero_division=0),
    }
    return {k: round(v, 4) for k, v in metrics.items()}


def run_all_baselines(train_data: Data, test_data: Data) -> Dict[str, Dict[str, float]]:
    """
    Run all heuristic baselines and return results.

    Returns:
        Dict mapping method name to metrics dict
    """
    results = {}
    for method in ["common_neighbors", "jaccard", "adamic_adar"]:
        print(f"  Evaluating {method}...")
        results[method] = evaluate_baseline(train_data, test_data, method)
        print(f"    AUC: {results[method]['auc_roc']:.4f} | "
              f"AP: {results[method]['avg_precision']:.4f} | "
              f"F1: {results[method]['f1']:.4f}")
    return results
