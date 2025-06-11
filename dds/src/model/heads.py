import torch
import torch.nn as nn
import numpy as np
import pandas as pd

class BinaryClassMLPv2Head(nn.Module):
    def __init__(self, input_dim, inner_dim, num_classes, actionvation_fn, pooler_dropout):
        super().__init__()

        self.num_classes = num_classes

        # Load and log-transform TPM matrix
        cell_feat_matrix = pd.read_csv('data/cell_tpm.csv', index_col=0).iloc[:, 1:].to_numpy()
        self.cell_feat_matrix = torch.FloatTensor(np.log1p(cell_feat_matrix))

        # Project cell features to 512 (fixed)
        self.mlp_cell = nn.Sequential(
            nn.Linear(self.cell_feat_matrix.shape[1], 512),
            nn.ReLU(),
            nn.Linear(512, 512),
        )

        self.dropout = nn.Dropout(pooler_dropout)

        # Inputs: 768 (head) + 768 (tail) + 512 (cell) = 2048
        self.fc1 = nn.Linear(2048, input_dim)
        self.fc2 = nn.Linear(input_dim, inner_dim)
        self.out = nn.Linear(inner_dim, 1)

        # Flag to prevent repeated logging
        self._printed_debug = False

    def forward(self, heads, tails, cell_lines):
        cell_lines_feats = self.cell_feat_matrix.to(heads.device)[cell_lines].type_as(heads)
        if cell_lines_feats.ndim == 1:
            cell_lines_feats = cell_lines_feats.unsqueeze(0)

        cell_lines_feats = self.mlp_cell(cell_lines_feats)

        # 🔍 Log shapes once to debug
        if not self._printed_debug:
            print("✅ heads shape:", heads.shape)
            print("✅ tails shape:", tails.shape)
            print("✅ cell_feats shape:", cell_lines_feats.shape)
            print("✅ concat shape (input to fc1):", torch.cat((heads, tails, cell_lines_feats), dim=1).shape)
            self._printed_debug = True

        xc = torch.cat((heads, tails, cell_lines_feats), dim=1)
        xc = self.fc1(xc)
        xc = torch.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = torch.relu(xc)
        xc = self.dropout(xc)
        return self.out(xc)
