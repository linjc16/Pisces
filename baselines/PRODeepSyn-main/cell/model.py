import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GraphConv


class GCNEncoder(nn.Module):

    def __init__(self, in_features: int, out_features: int, activation=F.relu, k: int = 2):
        super(GCNEncoder, self).__init__()
        assert k >= 2
        self.in_features = in_features
        self.out_features = out_features
        self.conv = nn.ModuleList()
        self.conv.append(GraphConv(in_features, 2 * out_features))
        self.activation = activation
        for _ in range(1, k - 1):
            self.conv.append(GraphConv(2 * out_features, 2 * out_features))
        self.conv.append(GraphConv(2 * out_features, out_features))

    def forward(self, g: dgl.DGLGraph, x: torch.Tensor):
        for conv in self.conv:
            x = self.activation(conv(g, x))
        return x


class Cell2Vec(nn.Module):

    def __init__(self, encoder: GCNEncoder, n_cell, n_dim):
        super(Cell2Vec, self).__init__()
        self.encoder = encoder
        self.embeddings = nn.Embedding(n_cell, n_dim)
        self.projector = nn.Sequential(
            nn.Linear(encoder.out_features, n_dim),
            nn.Dropout()
        )

    def forward(self, g: dgl.DGLGraph, x: torch.Tensor,
                x_indices: torch.LongTensor, c_indices: torch.LongTensor):
        encoded = self.encoder(g, x)
        encoded = encoded.index_select(0, x_indices)
        proj = self.projector(encoded).permute(1, 0)
        emb = self.embeddings(c_indices)
        out = torch.mm(emb, proj)
        return out


class RandomW(nn.Module):

    def __init__(self, n_node, n_node_dim, n_cell, n_dim):
        super(RandomW, self).__init__()
        self.encoder = nn.Embedding(n_node, n_node_dim)
        self.embeddings = nn.Embedding(n_cell, n_dim)
        self.projector = nn.Sequential(
            nn.Linear(n_node_dim, n_dim),
            nn.Dropout()
        )

    def forward(self, x_indices: torch.LongTensor, c_indices: torch.LongTensor):
        encoded = self.encoder(x_indices)
        proj = self.projector(encoded).permute(1, 0)
        emb = self.embeddings(c_indices)
        out = torch.mm(emb, proj)
        return out
