# GNN-Based Link Prediction on Twitch Social Networks

**Predicting Follower Connections Between Twitch Streamers Using Graph Neural Networks**

## Project Description

This project trains and compares three Graph Neural Network architectures (GCN, GraphSAGE, GAT) for link prediction on real Twitch social network data. Given a snapshot of the Twitch follower graph — where nodes are streamers and edges are follower relationships — the model predicts which pairs of streamers are likely to follow each other based on graph structure and gaming activity features.

### Deliverable 3 Improvements
- **Model refinement:** Added batch normalization and residual connections to all three GNN encoders for improved training stability and convergence
- **Heuristic baselines:** Implemented Common Neighbors, Jaccard Coefficient, and Adamic-Adar baselines to contextualize GNN performance
- **Cross-network experiments:** Train on one Twitch region, test on another to evaluate generalization
- **t-SNE visualization:** Interactive embedding visualization in the dashboard
- **D2 → D3 comparison:** Side-by-side metrics showing improvement from architectural refinements
- **Improved UI:** Cleaner layout with 5 tabs, baseline comparison, and embedding visualization

## Repository Structure

```
gnn-link-prediction/
├── src/
│   ├── dataset.py          # Data loading, Twitch regions, edge splitting
│   ├── models.py           # GCN, GraphSAGE, GAT with BatchNorm + residual
│   ├── train.py            # Training loop with early stopping
│   ├── evaluate.py         # AUC, AP, Precision, Recall, F1 metrics
│   ├── baselines.py        # Common Neighbors, Jaccard, Adamic-Adar
│   ├── cross_network.py    # Cross-region generalization experiment
│   └── utils.py            # Plotting and result export utilities
├── ui/
│   └── app.py              # Streamlit dashboard (v2)
├── notebooks/              # Jupyter notebooks
├── data/                   # Downloaded datasets (auto-populated)
├── results/                # Saved models, plots, metrics
├── docs/                   # Reports, diagrams, screenshots
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/yourusername/gnn-link-prediction.git
cd gnn-link-prediction
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## How to Run

### Command Line Training
```bash
python src/train.py --dataset TwitchEN --epochs 200 --models gcn sage gat
```

### Cross-Network Experiment
```bash
python src/cross_network.py --train-region DE --test-regions EN ES FR --arch gat
```

### Interactive Dashboard
```bash
streamlit run ui/app.py
```

The dashboard provides 5 tabs:
1. **Data Explorer** — Graph statistics and degree distribution
2. **Train** — Train GCN/GraphSAGE/GAT with configurable hyperparameters
3. **Results & D2↔D3** — Comparison tables, loss curves, ROC curves, D2 vs D3 improvement
4. **Baselines & Embeddings** — Heuristic baseline comparison + t-SNE visualization
5. **Predict** — Interactive single-pair and batch link prediction

## Current Results (D3)

Trained on TwitchEN (7,126 nodes, 77,774 edges) with BatchNorm + residual connections:

| Model     | AUC-ROC | Avg Precision | F1    | vs D2 AUC |
|-----------|---------|---------------|-------|-----------|
| GCN       | ~0.895  | ~0.912        | ~0.825| +0.013    |
| GraphSAGE | ~0.904  | ~0.919        | ~0.836| +0.013    |
| GAT       | ~0.915  | ~0.928        | ~0.845| +0.012    |

Heuristic baselines (same test set):

| Method           | AUC-ROC | F1    |
|------------------|---------|-------|
| Common Neighbors | ~0.72   | ~0.65 |
| Jaccard          | ~0.70   | ~0.62 |
| Adamic-Adar      | ~0.74   | ~0.67 |

GNNs outperform all heuristic baselines by 15-20% AUC, demonstrating the value of learned representations.

## Known Issues

- PyG's Twitch dataset loader returns 404; code auto-fallbacks to SNAP/GitHub
- MPS (Apple Silicon) may show minor numerical differences vs CUDA
- t-SNE computation takes ~10-30s depending on graph size
- Streamlit session state is lost on browser refresh

## Key Technologies

- **PyTorch** + **PyTorch Geometric** — GNN training and graph data
- **scikit-learn** — Metrics and t-SNE
- **Streamlit** — Interactive dashboard
- **Matplotlib / Seaborn** — Visualizations
- **SciPy** — Sparse adjacency for heuristic baselines

## Author

**Aryan Ghogare**
Semester Project — Applied Deep Learning, Spring 2026
Contact: [a.ghogare@ufl.edu]
