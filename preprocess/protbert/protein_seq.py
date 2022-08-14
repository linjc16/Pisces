import json
import pandas as pd
import os
import pdb
import numpy as np
import math
from tqdm import tqdm
from collections import defaultdict


gene_dir = 'data_feat/gene_new.json'
protein_seq_dir = 'data_feat/protein-v3.tsv'

with open(gene_dir, 'r+') as file:
    GENE_TO_STRING_DICT = file.read()

GENE_TO_STRING_DICT = json.loads(GENE_TO_STRING_DICT)
gene_list = list(GENE_TO_STRING_DICT.keys())

protein_seq_df = pd.read_csv(protein_seq_dir, sep='\t')
protein_seq_gene_col = protein_seq_df['Gene Names']
protein_seq_seq_col = protein_seq_df['Sequence']

PROTEIN_TO_INDEX_DICT = defaultdict(list)
GENE_STR_LEN_DICT = defaultdict(list) # for multiple strings

counts = 0

for i in tqdm(range(len(protein_seq_gene_col))):
    genes_curr = protein_seq_gene_col[i]
    # pdb.set_trace()
    try:
        if not isinstance(genes_curr, str) and np.isnan(genes_curr):
            continue
    except:
        pdb.set_trace()
    genes_curr = genes_curr.split(' ')
    for gene_curr in genes_curr:
        PROTEIN_TO_INDEX_DICT[gene_curr].append(i)
        GENE_STR_LEN_DICT[gene_curr].append(len(genes_curr))


# for key in PEOTEIN_TO_INDEX_DICT.keys():
#     if len(PEOTEIN_TO_INDEX_DICT[key]) > 1:
#         print(key, len(PEOTEIN_TO_INDEX_DICT[key]), GENE_STR_LEN_DICT[key])
#         counts += 1

protein_seqs = []
proteins = []
for gene in tqdm(gene_list):
    if gene not in GENE_STR_LEN_DICT.keys():
        print(f'gene {gene} not found')
        continue
    if 1 in GENE_STR_LEN_DICT[gene]:
        protein_seqs.append(protein_seq_seq_col[PROTEIN_TO_INDEX_DICT[gene][GENE_STR_LEN_DICT[gene].index(1)]])
    else:
        protein_seqs.append(protein_seq_seq_col[PROTEIN_TO_INDEX_DICT[gene][0]])
    proteins.append(gene)
protein_seq_df_new = pd.DataFrame.from_dict(
    {
        'Proteins': proteins,
        'Sequences': protein_seqs
    }
)

protein_seq_df_new.to_csv('data/protein_seq.csv', index=False)
# pdb.set_trace()