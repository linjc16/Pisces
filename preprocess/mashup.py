import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import pdb

with open('data/raw/string_human_genes.txt') as f:
    lines  = f.readlines()
    genes_ids = [line.replace('\n', '') for line in lines]


# genes_vec_mshup = np.loadtxt('data/raw/string_human_mashup_vectors_d800.txt')

# vec_dict = {}
genes_ids_set = set(genes_ids)

pdb.set_trace()

ppi_ids = pd.read_csv('baselines/PRODeepSyn-main/cell/data_ours/symbol2node.tsv', sep='\t')

for i in tqdm(range(len(ppi_ids))):
    sym_curr = ppi_ids['symbol'].iloc[i]
    if sym_curr not in genes_ids_set:
        print(f'{sym_curr} not found')
    