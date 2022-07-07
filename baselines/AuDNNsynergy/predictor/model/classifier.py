import torch
import torch.nn as nn


class DNN(nn.Module):
    def __init__(self, input_size, hidden_size, input_droupout=0.2, droupout=0.5):
        super(DNN, self).__init__()
        self.network = nn.Sequential(
            nn.Dropout(input_droupout),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(droupout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(droupout),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, drug1_feat: torch.Tensor, drug2_feat: torch.Tensor, cell_feat: torch.Tensor):
        feat = torch.cat([drug1_feat, drug2_feat, cell_feat], 1)
        out = self.network(feat)
        return out
