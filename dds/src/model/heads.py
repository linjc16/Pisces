from requests import head
from torch import layout, nn
from fairseq import utils
import torch
import torch.nn.functional as F
import numpy as np
import math

class BinaryClassMLPHead(nn.Module):
    
    def __init__(self,
                 input_dim,
                 inner_dim,
                 num_classes,
                 actionvation_fn,
                 pooler_dropout):

        super().__init__()
        
        self.num_classes = num_classes
        self.cell_emb = nn.Embedding(self.num_classes, inner_dim)
        nn.init.xavier_uniform_(self.cell_emb.weight)
        
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(actionvation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        
    def forward(self, heads, tails, cell_lines):

        cell_lines = self.cell_emb(cell_lines)
        cell_lines = F.normalize(cell_lines, dim=-1)

        x = heads + tails
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        
        scores = torch.matmul(cell_lines, x.unsqueeze(-1))

        return scores