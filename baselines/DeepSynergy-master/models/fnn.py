from turtle import forward
import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
import pdb

class DeepSynergy(nn.Module):
    def __init__(self, n_output=2, n_filters=32, embed_dim=128,num_features_xd=78, num_features_xt=37261, output_dim=128, dropout=0.1):
        
        super(DeepSynergy, self).__init__()

        self.activate_func = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.dropout_input = nn.Dropout(0.2)
        self.n_output = n_output

        # combined layers
        self.fc1 = nn.Linear((256 + 346 + 200) * 2 + num_features_xt, 8182)
        self.fc2 = nn.Linear(8182, 4096)
        self.out = nn.Linear(4096, self.n_output)

        
    def forward(self, data):
        x = self.fc1(data)
        x = self.activate_func(x)
        x = self.dropout_input(x)
        x = self.fc2(x)
        x = self.activate_func(x)
        x = self.dropout(x)
        x = self.out(x)

        # pdb.set_trace()

        return x