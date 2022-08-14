from torch import layout, nn
import torch
from fairseq import utils
import torch.nn.functional as F
import numpy as np
import math
import pandas as pd
import os
import networkx as nx
import collections
import pdb

# aux_data_dir = 'baselines/GraphSynergy-master/data_ours'

class DataPPI():
    def __init__(self, 
                 aux_data_dir,
                 n_hop=2, 
                 n_memory=32,):
        self.aux_data_dir = aux_data_dir
        self.n_hop = n_hop
        self.n_memory = n_memory

        self.ppi_df, self.cpi_df = self.load_data()

        self.node_map_dict, self.node_num_dict = self.get_node_map_dict()

        self.df_node_remap()

        self.graph = self.build_graph()

        self.cell_protein_dict = self.get_target_dict()

        self.cells = list(self.cell_protein_dict.keys())

        self.cell_neighbor_set = self.get_neighbor_set(items=self.cells,
                                                       item_target_dict=self.cell_protein_dict)

    def get_cell_neighbor_set(self):
        return self.cell_neighbor_set
    
    def get_node_num_dict(self):
        return self.node_num_dict
    
    def load_data(self):

        ppi_df = pd.read_excel(os.path.join(self.aux_data_dir, 'protein-protein_network.xlsx'))
        cpi_df = pd.read_csv(os.path.join(self.aux_data_dir, 'cell_protein.csv'))

        return ppi_df, cpi_df
    
    def get_node_map_dict(self):
        protein_node = list(set(self.ppi_df['protein_a']) | set(self.ppi_df['protein_b']))
        cell_tpm = pd.read_csv('data/cell_tpm.csv', index_col=0)
        cell_node = cell_tpm['cell_line_names'].tolist()


        node_num_dict = {'protein': len(protein_node), 'cell': len(cell_node)}
        mapping = {protein_node[idx]:idx for idx in range(len(protein_node))}
        mapping.update({cell_node[idx]:idx for idx in range(len(cell_node))})

        # display data info
        print('undirected graph')
        print('# proteins: {0}, # cells: {1}'.format(
                len(protein_node), len(cell_node)))
        print('# protein-protein interactions: {0}, # cell-protein associations: {1}'.format(
            len(self.ppi_df), len(self.cpi_df)))

        return mapping, node_num_dict

    def df_node_remap(self):
        self.ppi_df['protein_a'] = self.ppi_df['protein_a'].map(self.node_map_dict)
        self.ppi_df['protein_b'] = self.ppi_df['protein_b'].map(self.node_map_dict)
        self.ppi_df = self.ppi_df[['protein_a', 'protein_b']]

        self.cpi_df['cell'] = self.cpi_df['cell'].map(self.node_map_dict)
        self.cpi_df['protein'] = self.cpi_df['protein'].map(self.node_map_dict)
        self.cpi_df = self.cpi_df[['cell', 'protein']]


    def build_graph(self):
        tuples = [tuple(x) for x in self.ppi_df.values]
        graph = nx.Graph()
        graph.add_edges_from(tuples)
        return graph

    def get_target_dict(self):
        cp_dict = collections.defaultdict(list)
        cell_list = list(set(self.cpi_df['cell']))
        for cell in cell_list:
            cell_df = self.cpi_df[self.cpi_df['cell']==cell]
            target = list(set(cell_df['protein']))
            cp_dict[cell] = target

        return cp_dict

    def get_neighbor_set(self, items, item_target_dict):
        print('constructing neighbor set ...')

        neighbor_set = collections.defaultdict(list)
        for item in items:
            for hop in range(self.n_hop):
                # use the target directly
                if hop == 0:
                    replace = len(item_target_dict[item]) < self.n_memory
                    target_list = list(np.random.choice(item_target_dict[item], size=self.n_memory, replace=replace))
                else:
                    # use the last one to find k+1 hop neighbors
                    origin_nodes = neighbor_set[item][-1]
                    neighbors = []
                    for node in origin_nodes:
                        neighbors += self.graph.neighbors(node)
                    # sample
                    replace = len(neighbors) < self.n_memory
                    target_list = list(np.random.choice(neighbors, size=self.n_memory, replace=replace))
                
                neighbor_set[item].append(target_list)

        return neighbor_set

class BasePPIHead(nn.Module):

    def __init__(self):
        super().__init__()
        self.n_hop = 2
        ppi_loader = DataPPI(
            aux_data_dir='baselines/GraphSynergy-master/data_ours_3fold',
            n_hop=self.n_hop)

        self.cell_neighbor_set = ppi_loader.get_cell_neighbor_set()
        node_num_dict = ppi_loader.get_node_num_dict()
        self.protein_num = node_num_dict['protein']

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

    def _aggregation(self, item_i_list):
        # [batch_size, n_hop+1, emb_dim]
        item_i_concat = torch.cat(item_i_list, 1)
        # [batch_size, emb_dim]
        item_embeddings = self.aggregation_function(item_i_concat)
        return item_embeddings



class BinaryClassMLPPPIHead(nn.Module):
    
    def __init__(self,
                 input_dim,
                 inner_dim,
                 num_classes,
                 actionvation_fn,
                 pooler_dropout):

        super().__init__()
        
        self.cell_num = num_classes
        self.emb_dim = inner_dim
        self.n_hop = 2
        
        ppi_loader = DataPPI(
            aux_data_dir='baselines/GraphSynergy-master/data_ours_3fold',
            n_hop=self.n_hop)

        self.cell_neighbor_set = ppi_loader.get_cell_neighbor_set()
        node_num_dict = ppi_loader.get_node_num_dict()
        self.protein_num = node_num_dict['protein']

        self.protein_embedding = nn.Embedding(self.protein_num, self.emb_dim)
        self.cell_embedding = nn.Embedding(self.cell_num, self.emb_dim)
        self.aggregation_function = nn.Linear(self.emb_dim * self.n_hop, self.emb_dim)


        self.dropout = nn.Dropout(p=pooler_dropout)

        # combined layers
        self.fc1 = nn.Linear(2 * input_dim + inner_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, inner_dim)
        self.out = nn.Linear(inner_dim, 1)
        
    def forward(self, heads, tails, cells):

        cell_embeddings = self.cell_embedding(cells).squeeze(1)

        cells_neighbors = []
        for hop in range(self.n_hop):
            cells_neighbors.append(torch.LongTensor([self.cell_neighbor_set[c][hop] \
                                                       for c in cells.squeeze(1).cpu().numpy().tolist()]).to(heads.device))
        
        cell_neighbors_emb_list = self._get_neighbor_emb(cells_neighbors)
        cell_i_list = self._interaction_aggregation(cell_embeddings, cell_neighbors_emb_list)
        cell_embeddings = self._aggregation(cell_i_list)
        
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

    def _aggregation(self, item_i_list):
        # [batch_size, n_hop+1, emb_dim]
        item_i_concat = torch.cat(item_i_list, 1)
        # [batch_size, emb_dim]
        item_embeddings = self.aggregation_function(item_i_concat)
        return item_embeddings

class BinaryClassMLPPPIv2Head(nn.Module):
    
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

        
    def forward(self, heads, tails, cells):

        cells_neighbors = []
        for hop in range(self.n_hop):
            cells_neighbors.append(torch.LongTensor([self.cell_neighbor_set[c][hop] \
                                                       for c in cells.squeeze(1).cpu().numpy().tolist()]).to(heads.device))
        
        cell_neighbors_emb_list = self._get_neighbor_emb(cells_neighbors)
        cell_i_list = self._interaction_aggregation(cell_neighbors_emb_list)
        # pdb.set_trace()
        cell_embeddings = self._aggregation(cell_i_list)
        
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



class BinaryClassMLPOutInterPPIHead(nn.Module):
    
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
        
    def forward(self, heads, tails, cells):

        cells_neighbors = []
        for hop in range(self.n_hop):
            cells_neighbors.append(torch.LongTensor([self.cell_neighbor_set[c][hop] \
                                                       for c in cells.squeeze(1).cpu().numpy().tolist()]).to(heads.device))
        
        cell_neighbors_emb_list = self._get_neighbor_emb(cells_neighbors)
        cell_i_list = self._interaction_aggregation(cell_neighbors_emb_list)
        # pdb.set_trace()
        cell_embeddings = self._aggregation(cell_i_list)
        
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

class BinaryClassDVPPIMLPHead(nn.Module):
    
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


        self.dropout = nn.Dropout(p=pooler_dropout)

        # combined layers
        self.fc1 = nn.Linear(2 * input_dim + inner_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, inner_dim)
        self.out = nn.Linear(inner_dim, 1)

        self.mix_linear = nn.Linear(dv_input_dim + input_dim, input_dim)
    
    
    def forward(self, drug_a, dv_drug_a, drug_b, dv_drug_b, cells):

        cells_neighbors = []
        for hop in range(self.n_hop):
            cells_neighbors.append(torch.LongTensor([self.cell_neighbor_set[c][hop] \
                                                       for c in cells.squeeze(1).cpu().numpy().tolist()]).to(drug_a.device))
        
        cell_neighbors_emb_list = self._get_neighbor_emb(cells_neighbors)
        cell_i_list = self._interaction_aggregation(cell_neighbors_emb_list)
        # pdb.set_trace()
        cell_embeddings = self._aggregation(cell_i_list)
        
        heads = self.mix_linear(torch.cat([drug_a, dv_drug_a], dim=1))
        tails = self.mix_linear(torch.cat([drug_b, dv_drug_b], dim=1))
    
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

class BinaryClassMLPAttnPPIHead(nn.Module):
    
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

        

        self.linear_drug = nn.Linear(input_dim, inner_dim)
        self.linear_cell = nn.Linear(inner_dim, inner_dim)

        self.static_transform = nn.Sequential(
            nn.Linear(inner_dim, inner_dim),
            nn.Tanh()
        )

        self.k_matrix = nn.Linear(inner_dim, inner_dim)
        self.q_matrix = nn.Linear(inner_dim, inner_dim)
        self.v_matrix = nn.Linear(inner_dim, inner_dim)


        self.fc = nn.Linear(inner_dim, 1)

        
    def forward(self, heads, tails, cells):

        cells_neighbors = []
        for hop in range(self.n_hop):
            cells_neighbors.append(torch.LongTensor([self.cell_neighbor_set[c][hop] \
                                                       for c in cells.squeeze(1).cpu().numpy().tolist()]).to(heads.device))
        
        cell_neighbors_emb_list = self._get_neighbor_emb(cells_neighbors)
        cell_i_list = self._interaction_aggregation(cell_neighbors_emb_list)
        # pdb.set_trace()
        cell_embs = self._aggregation(cell_i_list)
        

        heads = self.linear_drug(heads)
        tails = self.linear_drug(tails)
        cell_embs = self.linear_cell(cell_embs)

        feat_seq = torch.cat([heads.unsqueeze(1), tails.unsqueeze(1), cell_embs.unsqueeze(1)], dim=1) # [N, 3, dim]

        # static
        feat_seq_static = self.static_transform(feat_seq)

        # dynamic
        k_feat_seq = self.k_matrix(feat_seq) # [N, 3, dim]
        q_feat_seq = self.q_matrix(feat_seq) # [N, 3, dim]
        v_feat_seq = self.v_matrix(feat_seq) # [N, 3, dim]

        dot_prdt = torch.bmm(q_feat_seq, k_feat_seq.transpose(1, 2)) # [N, 3, 3]
        dot_prdt.masked_fill_(
            torch.eye(3, 3, dtype=bool).repeat(dot_prdt.size(0), 1, 1).to(heads.device),
            float('-inf')
        )

        alpha = torch.softmax(dot_prdt, dim=2) # [N, 3, 3]

        feat_seq_dynamic = torch.tanh(torch.bmm(alpha, v_feat_seq)) # [N, 3, dim]
        

        out = self.fc((feat_seq_dynamic - feat_seq_static) ** 2) # [N, 3, 1]
        # out = torch.sigmoid(out).mean(dim=1)
        # pdb.set_trace()
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

class BinaryClassMLPPPIInnerMixHead(nn.Module):
    
    def __init__(self,
                 input_dim,
                 inner_dim,
                 num_classes,
                 actionvation_fn,
                 pooler_dropout):

        super().__init__()
        
        self.cell_num = num_classes
        self.emb_dim = inner_dim
        self.n_hop = 2
        
        ppi_loader = DataPPI(
            aux_data_dir='baselines/GraphSynergy-master/data_ours',
            n_hop=self.n_hop)

        self.cell_neighbor_set = ppi_loader.get_cell_neighbor_set()
        node_num_dict = ppi_loader.get_node_num_dict()
        self.protein_num = node_num_dict['protein']

        self.protein_embedding = nn.Embedding(self.protein_num, self.emb_dim)
        self.cell_embedding = nn.Embedding(self.cell_num, self.emb_dim)
        self.aggregation_function = nn.Linear(self.emb_dim * self.n_hop, self.emb_dim)


        self.dropout = nn.Dropout(p=pooler_dropout)

        # combined layers
        self.fc1 = nn.Linear(2 * input_dim + inner_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, inner_dim)
        self.out = nn.Linear(inner_dim, 1)
        
    def forward(self, heads, tails, cells):

        cell_embeddings = self.cell_embedding(cells).squeeze(1)

        cells_neighbors = []
        for hop in range(self.n_hop):
            cells_neighbors.append(torch.LongTensor([self.cell_neighbor_set[c][hop] \
                                                       for c in cells.squeeze(1).cpu().numpy().tolist()]).to(heads.device))
        
        cell_neighbors_emb_list = self._get_neighbor_emb(cells_neighbors)
        cell_i_list = self._interaction_aggregation(cell_embeddings, cell_neighbors_emb_list)
        cell_embeddings = self._aggregation(cell_i_list)
        
        # inner mix
        lamb_heads = torch.FloatTensor(np.random.beta(0.2, 0.2, heads.size(0))).to(heads.device).type_as(heads).unsqueeze(1)
        lamb_tails = torch.FloatTensor(np.random.beta(0.2, 0.2, heads.size(0))).to(heads.device).type_as(heads).unsqueeze(1)
        if np.random.random() >= 0.75:
            heads_new = heads * lamb_heads + tails * (1 - lamb_heads)
            tails_new = tails * lamb_tails + heads * (1 - lamb_tails)
        else:
            heads_new = heads
            tails_new = tails
        # concat
        xc = torch.cat((heads_new, tails_new, cell_embeddings), dim=1)
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

    def _aggregation(self, item_i_list):
        # [batch_size, n_hop+1, emb_dim]
        item_i_concat = torch.cat(item_i_list, 1)
        # [batch_size, emb_dim]
        item_embeddings = self.aggregation_function(item_i_concat)
        return item_embeddings

class BinaryClassMLPPPIInterMixHead(BasePPIHead):
    
    def __init__(self,
                 input_dim,
                 inner_dim,
                 num_classes,
                 actionvation_fn,
                 pooler_dropout):

        super().__init__()
        
        self.cell_num = num_classes
        self.emb_dim = inner_dim
        self.n_hop = 2
        
        ppi_loader = DataPPI(
            aux_data_dir='baselines/GraphSynergy-master/data_ours',
            n_hop=self.n_hop)

        self.cell_neighbor_set = ppi_loader.get_cell_neighbor_set()
        node_num_dict = ppi_loader.get_node_num_dict()
        self.protein_num = node_num_dict['protein']

        self.protein_embedding = nn.Embedding(self.protein_num, self.emb_dim)
        self.cell_embedding = nn.Embedding(self.cell_num, self.emb_dim)
        self.aggregation_function = nn.Linear(self.emb_dim * self.n_hop, self.emb_dim)


        self.dropout = nn.Dropout(p=pooler_dropout)

        # combined layers
        self.fc1 = nn.Linear(2 * input_dim + inner_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, inner_dim)
        self.out = nn.Linear(inner_dim, 1)
        
    def forward(self, heads, tails, cells, targets):

        cell_embeddings = self.cell_embedding(cells).squeeze(1)

        cells_neighbors = []
        for hop in range(self.n_hop):
            cells_neighbors.append(torch.LongTensor([self.cell_neighbor_set[c][hop] \
                                                       for c in cells.squeeze(1).cpu().numpy().tolist()]).to(heads.device))
        
        cell_neighbors_emb_list = self._get_neighbor_emb(cells_neighbors)
        cell_i_list = self._interaction_aggregation(cell_embeddings, cell_neighbors_emb_list)
        cell_embeddings = self._aggregation(cell_i_list)

        # inter mix
        lambs = torch.FloatTensor(np.random.beta(0.2, 0.2, heads.size(0) // 2)).to(heads.device).type_as(heads).unsqueeze(1)
        
        # concat
        xc = torch.cat((heads, tails, cell_embeddings), dim=1)

        # add some dense layers
        # pdb.set_trace()

        num_pairs = xc.size(0) // 2
        xc_1 = xc[0:num_pairs, :]
        xc_2 = xc[num_pairs:num_pairs * 2, :]
        xc = lambs * xc_1 + (1 - lambs) * xc_2

        targets = F.one_hot(targets.long().view(-1), num_classes=2).to(heads.device).type_as(heads)
        targets = lambs * targets[0:num_pairs, :] + (1 - lambs) * targets[num_pairs:num_pairs * 2, :].detach()
        
        
        xc = self.fc1(xc)
        xc = torch.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = torch.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)

        return out, targets

    def forward_eval(self, heads, tails, cells):

        cell_embeddings = self.cell_embedding(cells).squeeze(1)

        cells_neighbors = []
        for hop in range(self.n_hop):
            cells_neighbors.append(torch.LongTensor([self.cell_neighbor_set[c][hop] \
                                                       for c in cells.squeeze(1).cpu().numpy().tolist()]).to(heads.device))
        
        cell_neighbors_emb_list = self._get_neighbor_emb(cells_neighbors)
        cell_i_list = self._interaction_aggregation(cell_embeddings, cell_neighbors_emb_list)
        cell_embeddings = self._aggregation(cell_i_list)

        # concat
        xc = torch.cat((heads, tails, cell_embeddings), dim=1)
        xc = self.fc1(xc)
        xc = torch.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = torch.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)

        return out
if __name__ == '__main__':
    loader = DataPPI(aux_data_dir='baselines/GraphSynergy-master/data_ours')
    pdb.set_trace()
