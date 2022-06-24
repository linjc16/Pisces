import torch
import torch.nn as nn
import torch.nn.functional as F

from base import BaseModel

class GraphSynergy(BaseModel):
    def __init__(self, 
                 protein_num, 
                 cell_num,
                 drug_num,
                 emb_dim, 
                 n_hop,
                 l1_decay,
                 therapy_method):
        super(GraphSynergy, self).__init__()
        self.protein_num = protein_num
        self.cell_num = cell_num
        self.drug_num = drug_num
        self.emb_dim = emb_dim
        self.n_hop = n_hop
        self.l1_decay = l1_decay
        self.therapy_method = therapy_method

        self.protein_embedding = nn.Embedding(self.protein_num, self.emb_dim)
        self.cell_embedding = nn.Embedding(self.cell_num, self.emb_dim)
        self.drug_embedding = nn.Embedding(self.drug_num, self.emb_dim)
        self.aggregation_function = nn.Linear(self.emb_dim*self.n_hop, self.emb_dim)

        if self.therapy_method == 'transformation_matrix':
            self.combine_function = nn.Linear(self.emb_dim*2, self.emb_dim, bias=False)
        elif self.therapy_method == 'weighted_inner_product':
            self.combine_function = nn.Linear(in_features=2, out_features=1, bias=False)

    def forward(self,
                cells: torch.LongTensor,
                drug1: torch.LongTensor,
                drug2: torch.LongTensor,
                cell_neighbors: list,
                drug1_neighbors: list,
                drug2_neighbors: list):
        cell_embeddings = self.cell_embedding(cells)
        drug1_embeddings = self.drug_embedding(drug1)
        drug2_embeddings = self.drug_embedding(drug2)

        cell_neighbors_emb_list = self._get_neighbor_emb(cell_neighbors)
        drug1_neighbors_emb_list = self._get_neighbor_emb(drug1_neighbors)
        drug2_neighbors_emb_list = self._get_neighbor_emb(drug2_neighbors)
        # embedding loss
        emb_loss = self._emb_loss(cell_embeddings, drug1_embeddings, 
                                  drug2_embeddings, cell_neighbors_emb_list,
                                  drug1_neighbors_emb_list, drug2_neighbors_emb_list)

        cell_i_list = self._interaction_aggregation(cell_embeddings, cell_neighbors_emb_list)
        drug1_i_list = self._interaction_aggregation(drug1_embeddings, drug1_neighbors_emb_list)
        drug2_i_list = self._interaction_aggregation(drug2_embeddings, drug2_neighbors_emb_list)

        # [batch_size, dim]
        cell_embeddings = self._aggregation(cell_i_list)
        drug1_embeddings = self._aggregation(drug1_i_list)
        drug2_embeddings = self._aggregation(drug2_i_list)

        score = self._therapy(drug1_embeddings, drug2_embeddings, cell_embeddings) - \
                self._toxic(drug1_embeddings, drug2_embeddings)

        return score, emb_loss

    def _get_neighbor_emb(self, neighbors):
        neighbors_emb_list = []
        for hop in range(self.n_hop):
            neighbors_emb_list.append(self.protein_embedding(neighbors[hop]))
        return neighbors_emb_list

    def _interaction_aggregation(self, item_embeddings, neighbors_emb_list):
        interact_list = []
        for hop in range(self.n_hop):
            # [batch_size, n_memory, dim]
            neighbor_emb = neighbors_emb_list[hop]
            # [batch_size, dim, 1]
            item_embeddings_expanded = torch.unsqueeze(item_embeddings, dim=2)
            # [batch_size, n_memory]
            contributions = torch.squeeze(torch.matmul(neighbor_emb,
                                                       item_embeddings_expanded))
            # [batch_size, n_memory]
            contributions_normalized = F.softmax(contributions, dim=1)
            # [batch_size, n_memory, 1]
            contributions_expaned = torch.unsqueeze(contributions_normalized, dim=2)
            # [batch_size, dim]
            i = (neighbor_emb * contributions_expaned).sum(dim=1)
            # update item_embeddings
            item_embeddings = i
            interact_list.append(i)
        return interact_list

    def _therapy(self, drug1_embeddings, drug2_embeddings, cell_embeddings):
        if self.therapy_method == 'transformation_matrix':
            combined_durg = self.combine_function(torch.cat([drug1_embeddings, drug2_embeddings], dim=1))
            therapy_score = (combined_durg * cell_embeddings).sum(dim=1)
        elif self.therapy_method == 'weighted_inner_product':
            drug1_score = torch.unsqueeze((drug1_embeddings * cell_embeddings).sum(dim=1), dim=1)
            drug2_score = torch.unsqueeze((drug2_embeddings * cell_embeddings).sum(dim=1), dim=1)
            therapy_score = torch.squeeze(self.combine_function(torch.cat([drug1_score, drug2_score], dim=1)))
        elif self.therapy_method == 'max_pooling':
            combine_drug = torch.max(drug1_embeddings, drug2_embeddings)
            therapy_score = (combine_drug * cell_embeddings).sum(dim=1)
        return therapy_score

    def _toxic(self, drug1_embeddings, drug2_embeddings):
        return (drug1_embeddings * drug2_embeddings).sum(dim=1)

    def _aggregation(self, item_i_list):
        # [batch_size, n_hop+1, emb_dim]
        item_i_concat = torch.cat(item_i_list, 1)
        # [batch_size, emb_dim]
        item_embeddings = self.aggregation_function(item_i_concat)
        return item_embeddings

    def _emb_loss(self, cell_embeddings, drug1_embeddings, drug2_embeddings, 
                  cell_neighbors_emb_list, drug1_neighbors_emb_list, drug2_neighbors_emb_list):
        item_regularizer = (torch.norm(cell_embeddings) ** 2
                          + torch.norm(drug1_embeddings) ** 2
                          + torch.norm(drug2_embeddings) ** 2) / 2
        node_regularizer = 0
        for hop in range(self.n_hop):
            node_regularizer += (torch.norm(cell_neighbors_emb_list[hop]) ** 2
                              +  torch.norm(drug1_neighbors_emb_list[hop]) ** 2
                              +  torch.norm(drug2_neighbors_emb_list[hop]) ** 2) / 2
        
        emb_loss = self.l1_decay * (item_regularizer + node_regularizer) / cell_embeddings.shape[0]

        return emb_loss