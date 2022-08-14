from torch import nn
from fairseq import utils
import torch
import torch.nn.functional as F
import numpy as np
import math
import pandas as pd
import pdb
from transformers import BertModel, BertTokenizer
from tqdm import tqdm

class BinaryClassProtSeqFrozenMLPHead(nn.Module):
    
    def __init__(self,
                 input_dim,
                 inner_dim,
                 num_classes,
                 actionvation_fn,
                 pooler_dropout):

        super().__init__()
        
        self.num_classes = num_classes

        cell_feat_df = pd.read_csv('data/protein_sequence/cell_tpm_processed.csv')
        cell_feat_matrix = cell_feat_df.iloc[:, 1:].to_numpy()
        self.cell_feat_matrix = torch.FloatTensor(np.log2(cell_feat_matrix))
        # pdb.set_trace()
        # prot_seq_df = pd.read_csv('data/protein_sequence/protein_seq_processed.csv')

        # prot_seq = prot_seq_df['Sequences'].tolist()
        # pdb.set_trace()
        # assert prot_seq_df['Proteins'].tolist() == cell_feat_df.columns[1:].tolist()
        prot_feats = torch.load('data/protein_sequence/prot_feats.pt', map_location='cpu')

        self.prot_embds = nn.Embedding(prot_feats.size(0), prot_feats.size(1))
        self.prot_embds.weight.data.copy_(prot_feats)
        

        self.tpm_factor = nn.Embedding(1, self.cell_feat_matrix.size(1))
        nn.init.xavier_uniform_(self.tpm_factor.weight)
        self.tpm_bias = nn.Embedding(1, self.cell_feat_matrix.size(1))


        # self.mlp_prot = nn.Sequential(
        #     nn.Linear(prot_feats.size(1), prot_feats.size(1)),
        #     nn.ReLU(),
        #     nn.Linear(prot_feats.size(1), prot_feats.size(1)),
        #     nn.ReLU(),
        #     nn.Linear(prot_feats.size(1), prot_feats.size(1)),
        # )

        # self.mlp_cell = nn.Sequential(
        #     nn.Linear(self.prot_feats.size(1), input_dim),
        #     nn.ReLU(),
        #     nn.Linear(input_dim, input_dim),
        #     nn.ReLU(),
        #     nn.Linear(input_dim, inner_dim),
        #     nn.Tanh(),
        # )

        self.linear_drug = nn.Linear(input_dim, input_dim)

        self.dropout = nn.Dropout(p=pooler_dropout)

        # combined layers
        self.fc1 = nn.Linear(2 * input_dim + prot_feats.size(1), input_dim)
        self.fc2 = nn.Linear(input_dim, inner_dim)
        self.out = nn.Linear(inner_dim, 1)
        
    def forward(self, heads, tails, cell_lines):
        
        prot_feats = self.prot_embds.weight#.to(cell_lines.device) # [~20k, 1024]
        # prot_feats = self.mlp_prot(prot_feats)
        cell_feat_coefs_ori = self.cell_feat_matrix.to(cell_lines.device)[cell_lines, :].type_as(heads).squeeze(1)  # [N, ~20k]
        # coefs_temp = self.tpm_factor.weight * cell_feat_coefs_ori + self.tpm_bias.weight
        # coefs_temp.masked_fill_(
        #     (cell_feat_coefs_ori == 0),
        #     float('-inf')
        # )
        cell_feat_coefs = F.softmax(cell_feat_coefs_ori, dim=1) # [N, ~20k]
        # cell_feat_coefs = F.softmax(self.linear_tpm(cell_feat_coefs_ori), dim=1)
        # pdb.set_trace()

        cell_lines_feats = torch.matmul(cell_feat_coefs, prot_feats)
        # pdb.set_trace()
        # print(cell_lines_feats.std(dim=1).sum())
        cell_embeddings = cell_lines_feats #self.mlp_cell(cell_lines_feats)
        pdb.set_trace()
        # concat
        xc = torch.cat((self.linear_drug(heads), self.linear_drug(tails), cell_embeddings), dim=1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = torch.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = torch.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        
        return out