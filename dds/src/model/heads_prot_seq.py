import torch
import torch.nn as nn

class BinaryClassProtSeqFrozenMLPHead(nn.Module):
    def __init__(self, input_dim, inner_dim, num_classes, actionvation_fn='tanh', pooler_dropout=0.1):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation = getattr(torch.nn, actionvation_fn.capitalize())() if hasattr(torch.nn, actionvation_fn.capitalize()) else nn.Tanh()
        self.dropout = nn.Dropout(pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, x1, x2, cell_features):
        x = torch.cat([x1, x2, cell_features], dim=1)
        x = self.dropout(self.activation(self.dense(x)))
        return self.out_proj(x)
