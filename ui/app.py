"""
GNN Link Prediction Lab — Interactive Dashboard (v2)
Deliverable 3: Refined interface with baselines, D2 vs D3 comparison,
t-SNE embedding visualization, and improved UX.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import sys
import torch
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dataset import load_dataset, split_edges, get_dataset_stats, AVAILABLE_DATASETS
from models import build_model
from evaluate import evaluate_model, compute_predictions, compute_metrics, get_curve_data
from train import train_model, get_device
from baselines import run_all_baselines

# --- Page Config ---
st.set_page_config(page_title="GNN Link Prediction Lab", layout="wide", page_icon="🔗")

st.title("🔗 GNN Link Prediction Lab")
st.caption("Predict follower connections on Twitch social networks using Graph Neural Networks — v2.0")

# --- Session State ---
for key in ["trained_models", "data_loaded", "train_data", "results_history",
            "baseline_results", "full_data", "dataset_name", "val_data", "test_data"]:
    if key not in st.session_state:
        st.session_state[key] = {} if key in ("trained_models", "baseline_results") else (
            [] if key == "results_history" else (False if key == "data_loaded" else None))

# D2 reference results for comparison
D2_RESULTS = {
    "GCN": {"auc_roc": 0.882, "avg_precision": 0.899, "f1": 0.812},
    "SAGE": {"auc_roc": 0.891, "avg_precision": 0.908, "f1": 0.823},
    "GAT": {"auc_roc": 0.903, "avg_precision": 0.917, "f1": 0.834},
}

# --- Sidebar ---
st.sidebar.header("⚙️ Configuration")
dataset_name = st.sidebar.selectbox(
    "Dataset", list(AVAILABLE_DATASETS.keys()),
    index=list(AVAILABLE_DATASETS.keys()).index("TwitchEN"))
st.sidebar.caption(AVAILABLE_DATASETS[dataset_name])

st.sidebar.markdown("---")
st.sidebar.subheader("Model Settings")
architectures = st.sidebar.multiselect("GNN Architectures", ["gcn", "sage", "gat"], default=["gcn", "sage", "gat"])
epochs = st.sidebar.slider("Epochs", 50, 500, 200, step=50)
hidden_dim = st.sidebar.select_slider("Hidden Dim", [32, 64, 128, 256], value=128)
embed_dim = st.sidebar.select_slider("Embed Dim", [16, 32, 64, 128], value=64)
lr = st.sidebar.select_slider("Learning Rate", [0.001, 0.005, 0.01, 0.05], value=0.005)

st.sidebar.markdown("---")
st.sidebar.markdown("**v2.0** — Deliverable 3")
st.sidebar.markdown("Aryan Ghogare")

# --- Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Data Explorer", "🏋️ Train", "📈 Results & D2↔D3", "🧪 Baselines & Embeddings", "🔮 Predict",
])

COLORS = {"gcn": "#2196F3", "sage": "#4CAF50", "gat": "#FF5722"}

# ==================== TAB 1 ====================
with tab1:
    st.header("Dataset Explorer")
    if st.button("Load Dataset", key="load"):
        with st.spinner(f"Loading {dataset_name}..."):
            data = load_dataset(dataset_name)
            stats = get_dataset_stats(data)
            st.session_state.full_data = data
            st.session_state.data_stats = stats
            st.session_state.data_loaded = True
            st.session_state.dataset_name = dataset_name

    if st.session_state.data_loaded:
        stats = st.session_state.data_stats
        data = st.session_state.full_data
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Nodes", f"{stats['num_nodes']:,}")
        c2.metric("Edges", f"{stats['num_edges']:,}")
        c3.metric("Features", stats["num_features"])
        c4.metric("Avg Degree", f"{stats['avg_degree']:.1f}")
        c5.metric("Density", f"{stats['density']:.5f}")

        col1, col2 = st.columns(2)
        with col1:
            from torch_geometric.utils import degree
            deg = degree(data.edge_index[0], num_nodes=data.num_nodes).numpy()
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(deg, bins=50, color="#2196F3", edgecolor="white", alpha=0.85)
            ax.set_xlabel("Degree"); ax.set_ylabel("Count"); ax.set_title("Degree Distribution")
            ax.set_xlim(0, np.percentile(deg, 95))
            st.pyplot(fig); plt.close()
        with col2:
            st.subheader("Statistics")
            st.dataframe(pd.DataFrame({
                "Metric": ["Nodes", "Edges", "Features", "Avg Degree", "Max Degree",
                           "Median Degree", "Density", "% Degree < 5", "% Degree > 100"],
                "Value": [f"{stats['num_nodes']:,}", f"{stats['num_edges']:,}", stats["num_features"],
                          f"{stats['avg_degree']:.2f}", stats["max_degree"], f"{stats['median_degree']:.1f}",
                          f"{stats['density']:.6f}", f"{stats['pct_degree_lt_5']:.1f}%",
                          f"{stats['pct_degree_gt_100']:.1f}%"],
            }), hide_index=True, use_container_width=True)
    else:
        st.info("Click **Load Dataset** to begin.")

# ==================== TAB 2 ====================
with tab2:
    st.header("Train & Compare Models")
    if not st.session_state.data_loaded:
        st.warning("Load a dataset first in **Data Explorer**.")
    elif st.button("🚀 Start Training", type="primary"):
        data = st.session_state.full_data
        device = get_device()
        st.info(f"Device: **{device}** | Dataset: **{st.session_state.dataset_name}** | "
                f"Models: **{', '.join(a.upper() for a in architectures)}**")

        with st.spinner("Splitting edges..."):
            train_data, val_data, test_data = split_edges(data)
            st.session_state.train_data = train_data
            st.session_state.val_data = val_data
            st.session_state.test_data = test_data

        results = []
        progress = st.progress(0)
        for i, arch in enumerate(architectures):
            with st.spinner(f"Training {arch.upper()}..."):
                result = train_model(
                    arch=arch, train_data=train_data, val_data=val_data, test_data=test_data,
                    in_channels=data.num_node_features, hidden_channels=hidden_dim,
                    out_channels=embed_dim, lr=lr, epochs=epochs, device=device, save_dir="results")
                results.append(result)
                st.session_state.trained_models[arch] = result
            st.success(f"**{arch.upper()}** — AUC: {result['test_metrics']['auc_roc']:.4f} | "
                       f"AP: {result['test_metrics']['avg_precision']:.4f} | "
                       f"F1: {result['test_metrics']['f1']:.4f} | Time: {result['train_time']:.1f}s")
            progress.progress((i + 1) / len(architectures))
        st.session_state.results_history = results
        st.balloons()

# ==================== TAB 3 ====================
with tab3:
    st.header("Results & D2 ↔ D3 Comparison")
    if not st.session_state.results_history:
        st.info("Train models first.")
    else:
        results = st.session_state.results_history

        # D3 results table
        st.subheader("Deliverable 3 Results (with BatchNorm + Residual)")
        rows = []
        for r in results:
            rows.append({
                "Architecture": r["arch"].upper(), "Params": f"{r['total_params']:,}",
                "Best Epoch": r["best_epoch"], "Time (s)": f"{r['train_time']:.1f}",
                "AUC-ROC": f"{r['test_metrics']['auc_roc']:.4f}",
                "Avg Precision": f"{r['test_metrics']['avg_precision']:.4f}",
                "F1": f"{r['test_metrics']['f1']:.4f}",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        # D2 vs D3 comparison
        st.subheader("Improvement: D2 → D3")
        comp_rows = []
        for r in results:
            arch = r["arch"].upper()
            if arch in D2_RESULTS:
                d2 = D2_RESULTS[arch]
                d3 = r["test_metrics"]
                comp_rows.append({
                    "Model": arch,
                    "D2 AUC": f"{d2['auc_roc']:.4f}",
                    "D3 AUC": f"{d3['auc_roc']:.4f}",
                    "Δ AUC": f"{d3['auc_roc'] - d2['auc_roc']:+.4f}",
                    "D2 F1": f"{d2['f1']:.4f}",
                    "D3 F1": f"{d3['f1']:.4f}",
                    "Δ F1": f"{d3['f1'] - d2['f1']:+.4f}",
                })
        if comp_rows:
            st.dataframe(pd.DataFrame(comp_rows), hide_index=True, use_container_width=True)

        # Plots
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            for r in results:
                ax.plot(r["history"]["train_loss"], label=r["arch"].upper(),
                        color=COLORS.get(r["arch"], "#666"), linewidth=2)
            ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.legend(); ax.set_title("Training Loss")
            st.pyplot(fig); plt.close()
        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            for r in results:
                ax.plot(r["history"]["val_auc"], label=r["arch"].upper(),
                        color=COLORS.get(r["arch"], "#666"), linewidth=2)
            ax.set_xlabel("Epoch"); ax.set_ylabel("AUC-ROC"); ax.legend(); ax.set_title("Validation AUC")
            st.pyplot(fig); plt.close()

        # ROC
        st.subheader("ROC Curves (Test Set)")
        fig, ax = plt.subplots(figsize=(8, 5))
        for r in results:
            roc = r["curves"]["roc"]
            ax.plot(roc["fpr"], roc["tpr"], label=r["arch"].upper(),
                    color=COLORS.get(r["arch"], "#666"), linewidth=2)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend(); ax.set_title("ROC Curves")
        st.pyplot(fig); plt.close()

# ==================== TAB 4 ====================
with tab4:
    st.header("Baselines & Embedding Visualization")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Heuristic Baselines")
        if st.session_state.train_data is None:
            st.info("Train models first to compare against baselines.")
        elif st.button("Run Baselines"):
            with st.spinner("Computing heuristic baselines..."):
                bl_results = run_all_baselines(st.session_state.train_data, st.session_state.test_data)
                st.session_state.baseline_results = bl_results

        if st.session_state.baseline_results:
            bl = st.session_state.baseline_results
            bl_rows = []
            for method, metrics in bl.items():
                bl_rows.append({"Method": method.replace("_", " ").title(),
                                "AUC-ROC": f"{metrics['auc_roc']:.4f}",
                                "Avg Precision": f"{metrics['avg_precision']:.4f}",
                                "F1": f"{metrics['f1']:.4f}"})
            # Add GNN results for comparison
            for r in st.session_state.results_history:
                bl_rows.append({"Method": f"{r['arch'].upper()} (GNN)",
                                "AUC-ROC": f"{r['test_metrics']['auc_roc']:.4f}",
                                "Avg Precision": f"{r['test_metrics']['avg_precision']:.4f}",
                                "F1": f"{r['test_metrics']['f1']:.4f}"})
            st.dataframe(pd.DataFrame(bl_rows), hide_index=True, use_container_width=True)

    with col_right:
        st.subheader("t-SNE Embedding Visualization")
        if not st.session_state.trained_models:
            st.info("Train a model first.")
        else:
            sel = st.selectbox("Model for t-SNE", [k.upper() for k in st.session_state.trained_models])
            if st.button("Generate t-SNE"):
                arch = sel.lower()
                r = st.session_state.trained_models[arch]
                data = st.session_state.full_data
                device = get_device()

                model = build_model(arch, data.num_node_features, hidden_dim, embed_dim)
                model.load_state_dict(torch.load(r["model_path"], weights_only=True, map_location=device))
                model = model.to(device)
                model.eval()

                with st.spinner("Computing embeddings + t-SNE..."):
                    with torch.no_grad():
                        z = model.encode(data.x.to(device), data.edge_index.to(device)).cpu().numpy()

                    from sklearn.manifold import TSNE
                    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=500)
                    z_2d = tsne.fit_transform(z)

                    fig, ax = plt.subplots(figsize=(6, 5))
                    if hasattr(data, "y") and data.y is not None:
                        labels = data.y.cpu().numpy()
                        scatter = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=labels, cmap="Set1",
                                             s=5, alpha=0.6)
                        plt.colorbar(scatter, ax=ax, label="Class")
                    else:
                        ax.scatter(z_2d[:, 0], z_2d[:, 1], s=5, alpha=0.6, color="#2196F3")
                    ax.set_title(f"t-SNE of {sel} Embeddings")
                    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
                    st.pyplot(fig); plt.close()

# ==================== TAB 5 ====================
with tab5:
    st.header("Interactive Link Prediction")
    if not st.session_state.trained_models:
        st.info("Train models first.")
    else:
        available = list(st.session_state.trained_models.keys())
        sel_model = st.selectbox("Model", [m.upper() for m in available], key="pred_model")
        arch = sel_model.lower()
        data = st.session_state.full_data
        num_nodes = data.num_nodes

        st.markdown(f"Node IDs range from 0 to {num_nodes - 1}")

        col1, col2 = st.columns(2)
        node_a = col1.number_input("Node A", 0, num_nodes - 1, 0)
        node_b = col2.number_input("Node B", 0, num_nodes - 1, 1)

        if st.button("Predict Link"):
            r = st.session_state.trained_models[arch]
            device = get_device()
            model = build_model(arch, data.num_node_features, hidden_dim, embed_dim)
            model.load_state_dict(torch.load(r["model_path"], weights_only=True, map_location=device))
            model.to(device).eval()

            with torch.no_grad():
                z = model.encode(data.x.to(device), data.edge_index.to(device))
                edge_q = torch.tensor([[node_a], [node_b]], dtype=torch.long).to(device)
                prob = torch.sigmoid(model.decode(z, edge_q)).item()

            if prob > 0.7:
                st.success(f"**Link probability: {prob:.4f}** — Strong connection ✅")
            elif prob > 0.5:
                st.warning(f"**Link probability: {prob:.4f}** — Moderate connection")
            else:
                st.error(f"**Link probability: {prob:.4f}** — Weak/no connection ❌")

        st.markdown("---")
        st.subheader("Top Predicted Connections")
        target = st.number_input("Target Node", 0, num_nodes - 1, 0, key="batch_node")
        top_k = st.slider("Top K", 5, 50, 20)

        if st.button("Find Connections"):
            r = st.session_state.trained_models[arch]
            device = get_device()
            model = build_model(arch, data.num_node_features, hidden_dim, embed_dim)
            model.load_state_dict(torch.load(r["model_path"], weights_only=True, map_location=device))
            model.to(device).eval()

            with torch.no_grad():
                z = model.encode(data.x.to(device), data.edge_index.to(device))
                candidates = torch.arange(num_nodes, device=device)
                src = torch.full((num_nodes,), target, dtype=torch.long, device=device)
                logits = model.decode(z, torch.stack([src, candidates]))
                probs = torch.sigmoid(logits).cpu().numpy()

            edge_np = data.edge_index.numpy()
            existing = set(edge_np[1, edge_np[0] == target].tolist())

            pred_df = pd.DataFrame({"Node": range(num_nodes), "Probability": probs})
            pred_df = pred_df[pred_df["Node"] != target]
            pred_df["Existing"] = pred_df["Node"].apply(lambda n: "✅" if n in existing else "")
            pred_df = pred_df.sort_values("Probability", ascending=False).head(top_k)
            pred_df["Probability"] = pred_df["Probability"].round(4)
            st.dataframe(pred_df.reset_index(drop=True), hide_index=True, use_container_width=True)
