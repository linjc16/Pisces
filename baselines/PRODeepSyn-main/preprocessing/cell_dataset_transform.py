import os
import pandas as pd
import pdb
import numpy as np
from tqdm import tqdm

savedir = 'baselines/PRODeepSyn-main/cell/data_ours'

df_cell_feats = pd.read_csv('data/cell_read_count.csv', index_col=0)
symbol2node = pd.read_csv('baselines/PRODeepSyn-main/cell/data_ours/symbol2node.tsv', sep='\t')
df_gene_aliase = pd.read_csv('data/raw/9606.protein.aliases.v11.5.txt', sep='\t', names=['#string_protein_id', 'alias', 'source'], header=0)
ensp2node = pd.read_csv('baselines/PRODeepSyn-main/cell/data_ours/ensp2node.tsv', sep='\t')
string_protein_ids = df_gene_aliase['#string_protein_id']

cell_feats_matrix = df_cell_feats.iloc[:, 1:].to_numpy()
# nodes_ge
# pdb.set_trace()
nodes_ge = []
for gene in tqdm(df_cell_feats.keys()[1:]):
    index = symbol2node['node_id'][symbol2node['symbol'] == gene]
    if len(index) == 0:
        string_id = string_protein_ids[df_gene_aliase['alias'] == gene]
        if len(string_id) == 0:
            # print(f'gene {gene} not found')
            continue
        index = ensp2node['node_id'][ensp2node['ensp_id'] == string_id.iloc[0]]
        if len(index) == 0:
            # print(f'gene {gene} not found')
            continue
    try:
        nodes_ge.append(index.iloc[0])
    except:
        pdb.set_trace()

nodes_ge = np.array(nodes_ge)
np.save(os.path.join(savedir, 'nodes_ge.npy'), nodes_ge)
print(f'Save nodes_ge.npy, {nodes_ge.shape}')

target_ge = cell_feats_matrix[:, nodes_ge.tolist()]
np.save(os.path.join(savedir, 'target_ge.npy'), target_ge)
print(f'Save target_ge.npy, {target_ge.shape}')
# pdb.set_trace()