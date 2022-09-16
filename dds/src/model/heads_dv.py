from torch import layout, nn
from fairseq import utils
import torch
import torch.nn.functional as F
import numpy as np
import math
import pandas as pd
import pdb
from .heads_ppi import DataPPI



class BinaryClassDVPPIConsMLPHead(nn.Module):
    
    def __init__(self,
                 input_dim,
                 dv_input_dim,
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

        self.transformer_proj_head = nn.Linear(input_dim, inner_dim)
        self.graph_proj_head = nn.Linear(dv_input_dim, inner_dim)

        self.dropout = nn.Dropout(p=pooler_dropout)

        # combined layers
        self.fc1 = nn.Linear(3 * inner_dim, inner_dim * 2)
        self.fc2 = nn.Linear(inner_dim * 2, inner_dim)
        self.out = nn.Linear(inner_dim, 1)

        self.mix_linear = nn.Linear(inner_dim * 2, inner_dim)

        self.contrastive_loss = nn.CrossEntropyLoss(reduction='mean')
    
    def forward(self, drug_a, dv_drug_a, drug_b, dv_drug_b, cells):
        
        cells_neighbors = []
        for hop in range(self.n_hop):
            cells_neighbors.append(torch.LongTensor([self.cell_neighbor_set[c][hop] \
                                                       for c in cells.squeeze(1).cpu().numpy().tolist()]).to(drug_a.device))
        
        cell_neighbors_emb_list = self._get_neighbor_emb(cells_neighbors)
        cell_i_list = self._interaction_aggregation(cell_neighbors_emb_list)
        # pdb.set_trace()
        cell_embeddings = self._aggregation(cell_i_list)

        ta = self.transformer_proj_head(drug_a)
        tb = self.transformer_proj_head(drug_b)

        ga = self.graph_proj_head(dv_drug_a)
        gb = self.graph_proj_head(dv_drug_b)
        
        heads = self.mix_linear(torch.cat([ta, ga], dim=1))
        tails = self.mix_linear(torch.cat([tb, gb], dim=1))

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

        cosine_tga = self.get_cosine_loss(ta, ga)
        cosine_tgb = self.get_cosine_loss(tb, gb)

        return out, 0.5 * (cosine_tga + cosine_tgb)

    def get_cosine_loss(self, anchor, positive):
        anchor = anchor / torch.norm(anchor, dim=-1, keepdim=True)
        positive = positive / torch.norm(positive, dim=-1, keepdim=True)
        logits = torch.matmul(anchor, positive.T)
        logits = logits - torch.max(logits, 1, keepdim=True)[0].detach()
        targets = torch.arange(logits.shape[1]).long().to(logits.device)
        loss = self.contrastive_loss(logits, targets)
        return loss
    
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


class BinaryClassDVPPIConsMLPv4Head(nn.Module):
    
    def __init__(self,
                 input_dim,
                 dv_input_dim,
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

        self.transformer_proj_head = nn.Linear(input_dim, inner_dim)
        self.graph_proj_head = nn.Linear(dv_input_dim, inner_dim)
        self.mix_linear = nn.Linear(inner_dim * 2, inner_dim)
        self.layernorm_cell = nn.LayerNorm(inner_dim)

        
        self.classifier_1 = nn.Sequential(
            nn.Linear(3 * inner_dim, inner_dim * 2),
            nn.ReLU(),
            nn.Dropout(p=pooler_dropout),
        )
        
        self.classifier_2 = nn.Sequential(
            nn.Linear(inner_dim * 2, inner_dim),
            nn.ReLU(),
            nn.Dropout(p=pooler_dropout),
            nn.Linear(inner_dim, 1)
        )

        self.contrastive_loss = nn.CrossEntropyLoss(reduction='mean')
    
    def forward(self, drug_a, dv_drug_a, drug_b, dv_drug_b, cells, labels=None):

        cells_neighbors = []
        for hop in range(self.n_hop):
            cells_neighbors.append(torch.LongTensor([self.cell_neighbor_set[c][hop] \
                                                       for c in cells.squeeze(1).cpu().numpy().tolist()]).to(drug_a.device))
        
        cell_neighbors_emb_list = self._get_neighbor_emb(cells_neighbors)
        cell_i_list = self._interaction_aggregation(cell_neighbors_emb_list)

        cell_embeddings = self._aggregation(cell_i_list)
        cell_embeddings = self.layernorm_cell(cell_embeddings)


        ta = self.transformer_proj_head(drug_a)
        tb = self.transformer_proj_head(drug_b)

        ga = self.graph_proj_head(dv_drug_a)
        gb = self.graph_proj_head(dv_drug_b)
        
        heads = self.mix_linear(torch.cat([ta, ga], dim=1))
        tails = self.mix_linear(torch.cat([tb, gb], dim=1))


        # concat
        xc = torch.cat([heads, tails, cell_embeddings], dim=1)
        xc = self.classifier_1(xc)
        out = self.classifier_2(xc)

        combo_list = [[ta, tb], [ta, gb], [ga, tb], [ga, gb]]
        combo_idxes = np.random.choice(range(4), size=2, replace=False, p=[1/4] * 4)
        assert combo_idxes[0] != combo_idxes[1]

        xc_1_raw = torch.cat(combo_list[combo_idxes[0]] + [cell_embeddings], dim=1)
        xc_1 = self.classifier_1(xc_1_raw)
        sub_out_1 = self.classifier_2(xc_1)

        xc_2_raw = torch.cat(combo_list[combo_idxes[1]] + [cell_embeddings], dim=1)
        xc_2 = self.classifier_1(xc_2_raw)
        sub_out_2 = self.classifier_2(xc_2)

        cosine_ttgg = self.get_cosine_loss(xc_1_raw, xc_2_raw)
        consine = cosine_ttgg

        return out, consine, 0.5 * (sub_out_1 + sub_out_2)
    
    def get_cosine_loss(self, anchor, positive):
        anchor = anchor / torch.norm(anchor, dim=-1, keepdim=True)
        positive = positive / torch.norm(positive, dim=-1, keepdim=True)
        logits = torch.matmul(anchor, positive.T)
        logits = logits - torch.max(logits, 1, keepdim=True)[0].detach()
        targets = torch.arange(logits.shape[1]).long().to(logits.device)
        loss = self.contrastive_loss(logits, targets)
        return loss
    
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

