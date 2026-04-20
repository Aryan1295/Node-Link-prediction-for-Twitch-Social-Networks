"""
Dataset Loading and Edge Splitting
Handles downloading datasets and creating train/val/test edge splits
for the link prediction task.

Twitch data is loaded from SNAP or the original MUSAE GitHub repository,
bypassing PyG's broken graphmining.ai hosting.
"""

import os
import json
import urllib.request
import numpy as np
import torch
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.data import Data
from typing import Tuple, Dict, List


TWITCH_REGIONS = ["DE", "EN", "ES", "FR", "PT", "RU"]

# Region codes differ between MUSAE repo naming and PyG naming
# MUSAE uses: DE, ENGB, ES, FR, PTBR, RU
MUSAE_REGION_MAP = {
    "DE": "DE",
    "EN": "ENGB",
    "ES": "ES",
    "FR": "FR",
    "PT": "PTBR",
    "RU": "RU",
}

AVAILABLE_DATASETS = {
    "Cora": "Citation network (2.7K papers, 10.5K citations)",
    "CiteSeer": "Citation network (3.3K papers, 9.1K citations)",
    "TwitchDE": "Twitch Germany social network (~9.5K users, ~315K follows)",
    "TwitchEN": "Twitch English social network (~7.1K users, ~77K follows)",
    "TwitchES": "Twitch Spanish social network (~4.6K users, ~59K follows)",
    "TwitchFR": "Twitch French social network (~6.5K users, ~112K follows)",
    "TwitchPT": "Twitch Portuguese social network (~1.9K users, ~31K follows)",
    "TwitchRU": "Twitch Russian social network (~4.4K users, ~37K follows)",
}


def _download_file(url: str, path: str):
    """Download a file if it doesn't already exist."""
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"  Downloading {os.path.basename(path)}...")
    try:
        urllib.request.urlretrieve(url, path)
    except Exception as e:
        raise RuntimeError(f"Failed to download {url}: {e}")


def _try_download(urls: list, path: str):
    """Try multiple URLs in order until one succeeds."""
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for url in urls:
        try:
            print(f"  Trying {url}...")
            urllib.request.urlretrieve(url, path)
            return  # success
        except Exception:
            continue
    raise RuntimeError(
        f"Failed to download {os.path.basename(path)} from all sources.\n"
        f"Tried: {urls}\n"
        f"You can manually download from https://snap.stanford.edu/data/twitch-social-networks.html\n"
        f"and place the files in the data/twitch/ directory."
    )


def _load_twitch_manual(region: str, root: str = "data/") -> Data:
    """
    Load a Twitch regional network from SNAP or MUSAE GitHub.

    Downloads edges CSV, features JSON, and target CSV, then
    constructs a PyG Data object.

    Args:
        region: Two-letter region code (DE, EN, ES, FR, PT, RU)
        root: Base data directory

    Returns:
        PyG Data object with node features and edge index
    """
    musae_code = MUSAE_REGION_MAP[region]
    data_dir = os.path.join(root, "twitch", region)

    # File paths
    edges_path = os.path.join(data_dir, "edges.csv")
    features_path = os.path.join(data_dir, "features.json")
    target_path = os.path.join(data_dir, "target.csv")

    # Multiple source URLs for resilience
    # Source 1: SNAP (Stanford)
    # Source 2: MUSAE GitHub repo (original author)
    snap_base = f"https://snap.stanford.edu/data/twitch/{musae_code}"
    musae_base = f"https://raw.githubusercontent.com/benedekrozemberczki/MUSAE/master/input"

    _try_download(
        [f"{snap_base}/musae_{musae_code}_edges.csv",
         f"{musae_base}/edges/{musae_code}_edges.csv"],
        edges_path,
    )
    _try_download(
        [f"{snap_base}/musae_{musae_code}_features.json",
         f"{musae_base}/features/{musae_code}.json"],
        features_path,
    )
    _try_download(
        [f"{snap_base}/musae_{musae_code}_target.csv",
         f"{musae_base}/target/{musae_code}_target.csv"],
        target_path,
    )

    # --- Load edges ---
    edges = []
    with open(edges_path, "r") as f:
        header = next(f)  # skip header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                edges.append((int(parts[0]), int(parts[1])))

    if not edges:
        raise ValueError(f"No edges found for region {region}")

    src = [e[0] for e in edges]
    dst = [e[1] for e in edges]
    # Make undirected
    edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)

    # --- Load features (JSON: node_id -> list of feature indices) ---
    with open(features_path, "r") as f:
        features_dict = json.load(f)

    num_nodes = max(max(src), max(dst)) + 1
    # Find max feature index to determine feature dimension
    max_feat = 0
    for feat_list in features_dict.values():
        if feat_list:
            max_feat = max(max_feat, max(feat_list))
    num_features = max_feat + 1

    # Build binary feature matrix (multi-hot encoding of games played)
    x = torch.zeros(num_nodes, num_features, dtype=torch.float)
    for node_id_str, feat_list in features_dict.items():
        node_id = int(node_id_str)
        if node_id < num_nodes:
            for feat_idx in feat_list:
                x[node_id, feat_idx] = 1.0

    # --- Load target labels (mature content flag) ---
    y = torch.zeros(num_nodes, dtype=torch.long)
    try:
        with open(target_path, "r") as f:
            header = next(f)
            cols = header.strip().split(",")
            mature_idx = cols.index("mature") if "mature" in cols else -1
            if mature_idx >= 0:
                for line in f:
                    parts = line.strip().split(",")
                    node_id = int(parts[0])
                    if node_id < num_nodes:
                        y[node_id] = int(parts[mature_idx])
    except Exception:
        pass  # labels are optional for link prediction

    data = Data(x=x, edge_index=edge_index, y=y)
    return data


def load_dataset(name: str = "TwitchEN", root: str = "data/") -> Data:
    """
    Load a graph dataset by name.

    Args:
        name: Dataset name (Cora, CiteSeer, TwitchDE, TwitchEN, etc.)
        root: Directory to download/cache data

    Returns:
        PyG Data object with node features and edge index
    """
    if name in ("Cora", "CiteSeer"):
        from torch_geometric.datasets import Planetoid
        dataset = Planetoid(root=root, name=name)
        data = dataset[0]

    elif name.startswith("Twitch"):
        region = name.replace("Twitch", "")
        if region not in TWITCH_REGIONS:
            raise ValueError(f"Unknown Twitch region: {region}. Choose from {TWITCH_REGIONS}")

        # Skip PyG's broken loader, go straight to manual download
        print(f"Loading Twitch {region} from SNAP/GitHub...")
        data = _load_twitch_manual(region, root)

    else:
        raise ValueError(
            f"Unknown dataset: {name}.\nAvailable datasets:\n"
            + "\n".join(f"  {k}: {v}" for k, v in AVAILABLE_DATASETS.items())
        )

    print(f"\nDataset: {name}")
    print(f"  Nodes: {data.num_nodes:,}")
    print(f"  Edges: {data.num_edges:,}")
    print(f"  Features per node: {data.num_node_features}")
    print(f"  Avg degree: {data.num_edges / data.num_nodes:.1f}")

    if hasattr(data, "y") and data.y is not None:
        num_classes = data.y.unique().shape[0]
        print(f"  Node classes: {num_classes}")

    return data


def load_twitch_multi(regions: List[str] = None, root: str = "data/") -> Dict[str, Data]:
    """
    Load multiple Twitch regional graphs for cross-network experiments.

    Args:
        regions: List of region codes (default: all regions)
        root: Data directory

    Returns:
        Dict mapping region code to Data object
    """
    if regions is None:
        regions = TWITCH_REGIONS

    graphs = {}
    for region in regions:
        graphs[region] = load_dataset(f"Twitch{region}", root)

    return graphs


def split_edges(
    data: Data,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    neg_sampling_ratio: float = 1.0,
    seed: int = 42,
) -> Tuple[Data, Data, Data]:
    """
    Split edges into train/validation/test sets for link prediction.

    Uses PyG's RandomLinkSplit which:
    - Removes val/test edges from the training graph
    - Generates negative samples (non-edges) for evaluation
    - Preserves node features across all splits

    Args:
        data: Full graph data
        val_ratio: Fraction of edges for validation
        test_ratio: Fraction of edges for testing
        neg_sampling_ratio: Ratio of negative to positive samples
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    torch.manual_seed(seed)

    transform = RandomLinkSplit(
        num_val=val_ratio,
        num_test=test_ratio,
        is_undirected=True,
        add_negative_train_samples=True,
        neg_sampling_ratio=neg_sampling_ratio,
    )

    train_data, val_data, test_data = transform(data)

    print(f"\nEdge split (val={val_ratio}, test={test_ratio}):")
    print(f"  Train edges: {train_data.edge_label.sum().int().item():,} pos, "
          f"{(train_data.edge_label == 0).sum().item():,} neg")
    print(f"  Val edges:   {val_data.edge_label.sum().int().item():,} pos, "
          f"{(val_data.edge_label == 0).sum().item():,} neg")
    print(f"  Test edges:  {test_data.edge_label.sum().int().item():,} pos, "
          f"{(test_data.edge_label == 0).sum().item():,} neg")

    return train_data, val_data, test_data


def get_dataset_stats(data: Data) -> Dict:
    """Compute basic graph statistics for analysis."""
    from torch_geometric.utils import degree

    deg = degree(data.edge_index[0], num_nodes=data.num_nodes)

    stats = {
        "num_nodes": data.num_nodes,
        "num_edges": data.num_edges,
        "num_features": data.num_node_features,
        "avg_degree": round(deg.mean().item(), 2),
        "max_degree": int(deg.max().item()),
        "min_degree": int(deg.min().item()),
        "std_degree": round(deg.std().item(), 2),
        "median_degree": round(deg.median().item(), 2),
        "density": round(data.num_edges / (data.num_nodes * (data.num_nodes - 1)), 6),
    }

    deg_np = deg.numpy()
    stats["pct_degree_lt_5"] = round((deg_np < 5).mean() * 100, 1)
    stats["pct_degree_lt_20"] = round((deg_np < 20).mean() * 100, 1)
    stats["pct_degree_gt_100"] = round((deg_np > 100).mean() * 100, 1)

    return stats
