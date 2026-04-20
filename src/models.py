"""
GNN Models for Link Prediction
Implements GCN, GraphSAGE, and GAT encoders with a shared link prediction decoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv


class LinkPredictor(nn.Module):
    """
    Shared decoder that scores candidate edges from node embeddings.

    Takes a pair of node embeddings and predicts the probability
    of an edge existing between them.
    """

    def __init__(self, in_channels: int, hidden_channels: int = 64, dropout: float = 0.3):
        super().__init__()
        self.lin1 = nn.Linear(in_channels * 2, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, 1)
        self.dropout = dropout

    def forward(self, z_src: torch.Tensor, z_dst: torch.Tensor) -> torch.Tensor:
        """
        Predict edge probability for (src, dst) pairs.

        Args:
            z_src: Source node embeddings [num_edges, embed_dim]
            z_dst: Destination node embeddings [num_edges, embed_dim]

        Returns:
            Edge probabilities [num_edges]
        """
        # Concatenate source and destination embeddings
        edge_feat = torch.cat([z_src, z_dst], dim=-1)
        edge_feat = F.relu(self.lin1(edge_feat))
        edge_feat = F.dropout(edge_feat, p=self.dropout, training=self.training)
        return self.lin2(edge_feat).squeeze(-1)


class GCNEncoder(nn.Module):
    """
    Graph Convolutional Network encoder.

    Uses symmetric normalized aggregation with batch normalization
    and residual connections for improved training stability.
    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs[:-1]):
            h = conv(x, edge_index)
            h = self.bns[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            # Residual connection (if dimensions match)
            if h.shape == x.shape:
                h = h + x
            x = h
        x = self.convs[-1](x, edge_index)
        return x


class SAGEEncoder(nn.Module):
    """
    GraphSAGE encoder with batch normalization and residual connections.

    Samples and aggregates features from a node's neighbors,
    then concatenates with the node's own features.
    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs[:-1]):
            h = conv(x, edge_index)
            h = self.bns[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            if h.shape == x.shape:
                h = h + x
            x = h
        x = self.convs[-1](x, edge_index)
        return x


class GATEncoder(nn.Module):
    """
    Graph Attention Network encoder with batch normalization.

    Uses multi-head attention to learn importance weights for each
    neighbor, with batch norm for training stability.
    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 num_layers: int = 2, heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout))
        self.bns.append(nn.BatchNorm1d(hidden_channels * heads))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout))
            self.bns.append(nn.BatchNorm1d(hidden_channels * heads))
        self.convs.append(GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout))
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs[:-1]):
            h = conv(x, edge_index)
            h = self.bns[i](h)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            if h.shape == x.shape:
                h = h + x
            x = h
        x = self.convs[-1](x, edge_index)
        return x


class LinkPredictionModel(nn.Module):
    """
    Full link prediction model = GNN Encoder + Link Decoder.

    The encoder produces node embeddings, and the decoder scores
    candidate edges based on pairs of embeddings.
    """

    def __init__(self, encoder: nn.Module, embed_dim: int, decoder_hidden: int = 64,
                 decoder_dropout: float = 0.3):
        super().__init__()
        self.encoder = encoder
        self.decoder = LinkPredictor(embed_dim, decoder_hidden, decoder_dropout)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Get node embeddings from the encoder."""
        return self.encoder(x, edge_index)

    def decode(self, z: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
        """Score candidate edges given node embeddings."""
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        return self.decoder(src, dst)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_label_index: torch.Tensor) -> torch.Tensor:
        """Full forward pass: encode nodes, then score candidate edges."""
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)


def build_model(
    arch: str,
    in_channels: int,
    hidden_channels: int = 128,
    out_channels: int = 64,
    num_layers: int = 2,
    heads: int = 4,
    dropout: float = 0.3,
) -> LinkPredictionModel:
    """
    Factory function to build a link prediction model.

    Args:
        arch: Architecture name — 'gcn', 'sage', or 'gat'
        in_channels: Number of input node features
        hidden_channels: Hidden layer dimension
        out_channels: Output embedding dimension
        num_layers: Number of GNN layers
        heads: Number of attention heads (GAT only)
        dropout: Dropout probability

    Returns:
        LinkPredictionModel with the specified encoder
    """
    arch = arch.lower()
    if arch == "gcn":
        encoder = GCNEncoder(in_channels, hidden_channels, out_channels, num_layers, dropout)
    elif arch == "sage":
        encoder = SAGEEncoder(in_channels, hidden_channels, out_channels, num_layers, dropout)
    elif arch == "gat":
        encoder = GATEncoder(in_channels, hidden_channels, out_channels, num_layers, heads, dropout)
    else:
        raise ValueError(f"Unknown architecture: {arch}. Choose from: gcn, sage, gat")

    model = LinkPredictionModel(encoder, embed_dim=out_channels)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Built {arch.upper()} model: {total_params:,} parameters")

    return model
