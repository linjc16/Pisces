from torch import layout, nn
from fairseq import utils
import torch
import torch.nn.functional as F
import numpy as np
import math
import pandas as pd
import pdb

class BinaryClassMLPHead(nn.Module):
    
    def __init__(self,
                 input_dim,
                 inner_dim,
                 num_classes,
                 actionvation_fn,
                 pooler_dropout):

        super().__init__()
        
        self.num_classes = num_classes

        cell_feat_matrix = pd.read_csv('data/cell_tpm.csv', index_col=0).iloc[:, 1:].to_numpy()
        self.cell_feat_matrix = torch.FloatTensor(np.log2(cell_feat_matrix + 1e-3))
        # pdb.set_trace()
        # self.cell_feat_matrix = torch.FloatTensor(cell_feat_matrix)

        self.mlp_cell = nn.Sequential(
            nn.Linear(self.cell_feat_matrix.size(1), input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, inner_dim),
            nn.Tanh(),
        )
        
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(actionvation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        
    def forward(self, heads, tails, cell_lines):
        
        cell_lines_feats = self.cell_feat_matrix.to(cell_lines.device)[cell_lines, :].type_as(heads)
        cell_lines_feats = F.normalize(cell_lines_feats, 2, 1)
        cell_lines_feats = self.mlp_cell(cell_lines_feats)

        x = heads + tails
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        
        scores = torch.matmul(cell_lines_feats, x.unsqueeze(-1))

        return scores


class BinaryClassMLPv2Head(nn.Module):
    
    def __init__(self,
                 input_dim,
                 inner_dim,
                 num_classes,
                 actionvation_fn,
                 pooler_dropout):

        super().__init__()
        
        self.num_classes = num_classes

        cell_feat_matrix = pd.read_csv('data/cell_tpm.csv', index_col=0).iloc[:, 1:].to_numpy()
        # self.cell_feat_matrix = torch.FloatTensor(np.log2(cell_feat_matrix + 1e-3))
        self.cell_feat_matrix = torch.FloatTensor(np.log1p(cell_feat_matrix))
        # pdb.set_trace()
        # self.cell_feat_matrix = torch.FloatTensor(cell_feat_matrix)

        self.mlp_cell = nn.Sequential(
            nn.Linear(self.cell_feat_matrix.size(1), input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, inner_dim),
        )
        
        # self.dense = nn.Linear(input_dim, inner_dim)
        # self.activation_fn = utils.get_activation_fn(actionvation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)

        # combined layers
        self.fc1 = nn.Linear(2 * input_dim + inner_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, inner_dim)
        self.out = nn.Linear(inner_dim, 1)
        
    def forward(self, heads, tails, cell_lines):
        
        cell_lines_feats = self.cell_feat_matrix.to(cell_lines.device)[cell_lines, :].type_as(heads)
        # cell_lines_feats = F.normalize(cell_lines_feats, 2, 1)
        cell_lines_feats = self.mlp_cell(cell_lines_feats)
        
        # concat
        xc = torch.cat((heads, tails, cell_lines_feats.squeeze(1)), dim=1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = torch.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = torch.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)

        return out

class BinaryClassMLPv2NonormHead(nn.Module):
    
    def __init__(self,
                 input_dim,
                 inner_dim,
                 num_classes,
                 actionvation_fn,
                 pooler_dropout):

        super().__init__()
        
        self.num_classes = num_classes

        cell_feat_matrix = pd.read_csv('data/cell_tpm.csv', index_col=0).iloc[:, 1:].to_numpy()
        self.cell_feat_matrix = torch.FloatTensor(np.log2(cell_feat_matrix + 1e-3))
        # pdb.set_trace()
        # self.cell_feat_matrix = torch.FloatTensor(cell_feat_matrix)

        self.mlp_cell = nn.Sequential(
            nn.Linear(self.cell_feat_matrix.size(1), input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, inner_dim),
        )
        
        # self.dense = nn.Linear(input_dim, inner_dim)
        # self.activation_fn = utils.get_activation_fn(actionvation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)

        # combined layers
        self.fc1 = nn.Linear(2 * input_dim + inner_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, inner_dim)
        self.out = nn.Linear(inner_dim, 1)
        
    def forward(self, heads, tails, cell_lines):
        
        cell_lines_feats = self.cell_feat_matrix.to(cell_lines.device)[cell_lines, :].type_as(heads)
        cell_lines_feats = self.mlp_cell(cell_lines_feats)

        # concat
        xc = torch.cat((heads, tails, cell_lines_feats.squeeze(1)), dim=1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = torch.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = torch.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)

        return out

class BinaryClassMLPv3Head(nn.Module):
    
    def __init__(self,
                 input_dim,
                 inner_dim,
                 num_classes,
                 actionvation_fn,
                 pooler_dropout):

        super().__init__()
        
        self.num_classes = num_classes

        # cell_feat_matrix = pd.read_csv('data/cell_tpm.csv', index_col=0).iloc[:, 1:].to_numpy()
        # self.cell_feat_matrix = torch.FloatTensor(np.log2(cell_feat_matrix + 1e-3))

        cell_feat_matrix = np.load('baselines/PRODeepSyn-main/cell/data_ours/cell_feat.npy')
        self.cell_feat_matrix = torch.FloatTensor(cell_feat_matrix)
        # pdb.set_trace()
        # self.cell_feat_matrix = torch.FloatTensor(cell_feat_matrix)

        
        # self.dense = nn.Linear(input_dim, inner_dim)
        # self.activation_fn = utils.get_activation_fn(actionvation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)

        # combined layers
        self.fc1 = nn.Linear(2 * input_dim + inner_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, inner_dim)
        self.out = nn.Linear(inner_dim, 1)
        
    def forward(self, heads, tails, cell_lines):
        
        cell_lines_feats = self.cell_feat_matrix.to(cell_lines.device)[cell_lines, :].type_as(heads)
        # cell_lines_feats = F.normalize(cell_lines_feats, 2, 1)
        
        # concat
        xc = torch.cat((heads, tails, cell_lines_feats.squeeze(1)), dim=1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = torch.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = torch.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)

        return out

class BinaryClassMLPEmbHead(nn.Module):
    
    def __init__(self,
                 input_dim,
                 inner_dim,
                 num_classes,
                 actionvation_fn,
                 pooler_dropout):

        super().__init__()
        
        self.num_classes = num_classes

        self.cell_feat_matrix = nn.Embedding(num_classes, inner_dim)
        nn.init.xavier_uniform_(self.cell_feat_matrix.weight)
        # pdb.set_trace()
        # self.cell_feat_matrix = torch.FloatTensor(cell_feat_matrix)
        
        # self.dense = nn.Linear(input_dim, inner_dim)
        # self.activation_fn = utils.get_activation_fn(actionvation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)

        # combined layers
        self.fc1 = nn.Linear(2 * input_dim + inner_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, inner_dim)
        self.out = nn.Linear(inner_dim, 1)
        
    def forward(self, heads, tails, cell_lines):
        
        cell_lines_feats = self.cell_feat_matrix(cell_lines)
        # pdb.set_trace()
        # cell_lines_feats = F.normalize(cell_lines_feats, dim=-1)
        # cell_lines_feats = self.mlp_cell(cell_lines_feats)

        # concat
        xc = torch.cat((heads, tails, cell_lines_feats.squeeze(1)), dim=1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = torch.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = torch.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)

        return out

class BinaryClassMLPSimCLRHead(nn.Module):
    
    def __init__(self,
                 input_dim,
                 inner_dim,
                 num_classes,
                 actionvation_fn,
                 pooler_dropout):

        super().__init__()
        
        self.num_classes = num_classes

        cell_feat_matrix = pd.read_csv('data/cell_tpm.csv', index_col=0).iloc[:, 1:].to_numpy()
        self.cell_feat_matrix = torch.FloatTensor(np.log2(cell_feat_matrix + 1e-3))

        self.mlp_cell = nn.Sequential(
            nn.Linear(self.cell_feat_matrix.size(1), input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, inner_dim),
        )
        

        self.encoder = nn.Sequential(
            nn.Linear(2 * input_dim + inner_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, inner_dim),
            nn.ReLU(),
            nn.Linear(inner_dim, inner_dim)
        )
        # combined layers
        self.classifier = nn.Sequential(
            nn.Linear(inner_dim, inner_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=pooler_dropout),
            nn.Linear(inner_dim // 2, inner_dim // 4),
            nn.ReLU(),
            nn.Dropout(p=pooler_dropout),
            nn.Linear(inner_dim // 4, 1)
        )

        self.projector = nn.Linear(inner_dim, inner_dim)
        
    def forward(self, heads, tails, cell_lines):
        
        cell_lines_feats = self.cell_feat_matrix.to(cell_lines.device)[cell_lines, :].type_as(heads)
        cell_lines_feats = F.normalize(cell_lines_feats, 2, 1)
        cell_lines_feats = self.mlp_cell(cell_lines_feats)

        xc = torch.cat((heads, tails, cell_lines_feats.squeeze(1)), dim=1)

        h = self.encoder(xc)
        z = self.projector(h)

        out = self.classifier(h)
        

        return out, z, h