from torch import layout, nn
from fairseq import utils
import torch
import torch.nn.functional as F
import numpy as np
import math
import pandas as pd
import pdb
from .heads_ppi import DataPPI

class BinaryClassMLPPairHead(nn.Module):
    
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
        
        self.dropout = nn.Dropout(p=pooler_dropout)

        self.linear_drug = nn.Linear(input_dim, input_dim)

        # combined layers
        self.fc1 = nn.Linear(2 * input_dim + inner_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, inner_dim)
        self.out = nn.Linear(inner_dim, 1)
        
    def forward(self, heads, tails, cell_lines):
        
        cell_lines_feats = self.cell_feat_matrix.to(cell_lines.device)[cell_lines, :].type_as(heads)
        # cell_lines_feats = F.normalize(cell_lines_feats, 2, 1)
        cell_lines_feats = self.mlp_cell(cell_lines_feats)
        
        heads, tails = self.linear_drug(heads), self.linear_drug(tails)
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


class BinaryClassMLPPPIv2PairHead(nn.Module):
    
    def __init__(self,
                 input_dim,
                 inner_dim,
                 num_classes,
                 actionvation_fn,
                 pooler_dropout,
                 n_memory):

        super().__init__()
        
        self.cell_num = num_classes
        self.emb_dim = inner_dim
        self.n_hop = 2
        self.n_memory = n_memory
        
        ppi_loader = DataPPI(
            aux_data_dir='baselines/GraphSynergy-master/data_ours_3fold',
            n_hop=self.n_hop,
            n_memory=self.n_memory)

        self.cell_neighbor_set = ppi_loader.get_cell_neighbor_set()
        node_num_dict = ppi_loader.get_node_num_dict()
        self.protein_num = node_num_dict['protein']

        self.protein_embedding = nn.Embedding(self.protein_num, self.emb_dim)
        self.aggregation_function = nn.Linear(self.emb_dim * 2 * self.n_hop, self.emb_dim)

        
        self.dropout = nn.Dropout(p=pooler_dropout)

        # combined layers
        self.fc1 = nn.Linear(2 * input_dim + inner_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, inner_dim)
        self.out = nn.Linear(inner_dim, 1)

        self.linear_drug = nn.Linear(input_dim, input_dim)

        self.layernorm_drug = nn.LayerNorm(input_dim)
        self.layernorm_cell = nn.LayerNorm(inner_dim)

    def forward(self, heads, tails, cells):

        cells_neighbors = []
        for hop in range(self.n_hop):
            cells_neighbors.append(torch.LongTensor([self.cell_neighbor_set[c][hop] \
                                                       for c in cells.squeeze(1).cpu().numpy().tolist()]).to(heads.device))
        
        cell_neighbors_emb_list = self._get_neighbor_emb(cells_neighbors)
        cell_i_list = self._interaction_aggregation(cell_neighbors_emb_list)
        # pdb.set_trace()
        cell_embeddings = self._aggregation(cell_i_list)
        
        heads, tails = self.linear_drug(heads), self.linear_drug(tails)
        heads, tails = self.layernorm_drug(heads), self.layernorm_drug(tails)
        cell_embeddings = self.layernorm_cell(cell_embeddings)
        
        # concat
        xc = torch.cat((heads, tails, cell_embeddings), dim=1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = torch.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = torch.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)

        return out

    def _get_neighbor_emb(self, neighbors):
        neighbors_emb_list = []
        for hop in range(self.n_hop):
            neighbors_emb_list.append(self.protein_embedding(neighbors[hop]))
        return neighbors_emb_list

    def _interaction_aggregation(self, neighbors_emb_list):
        interact_list = []
        for hop in range(self.n_hop):
            # [batch_size, n_memory, dim]
            neighbor_emb = neighbors_emb_list[hop]
            aggr_mean = torch.mean(neighbor_emb, dim=1)
            aggr_max = torch.max(neighbor_emb, dim=1).values
            interact_list.append(torch.cat([aggr_mean, aggr_max], dim=-1))
        
        return interact_list

    def _aggregation(self, item_i_list):
        # [batch_size, n_hop+1, emb_dim]
        item_i_concat = torch.cat(item_i_list, 1)
        # [batch_size, emb_dim]
        item_embeddings = self.aggregation_function(item_i_concat)
        return item_embeddings