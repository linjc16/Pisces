import pandas as pd
import pdb
import json
import os
import numpy as np

drug_target_ori = pd.read_csv('data/raw/41467_2019_9186_MOESM4_ESM.csv')
gene_info = pd.read_csv('data/raw/gene_identifiers_20191101.csv')
aliases_df = pd.read_csv('data/raw/9606.protein.aliases.v11.5.txt', sep='\t', names=['#string_protein_id', 'alias', 'source'], header=0)

drug_smiles_df = pd.read_csv('data/drug_smiles.csv', index_col=0)

ensp2node_df = pd.read_csv('baselines/PRODeepSyn-main/cell/data/ensp2node.tsv', sep='\t')
ENSP_TO_NODE_DICT = {ensp: node for ensp, node in zip(ensp2node_df['ensp_id'].tolist(), ensp2node_df['node_id'].tolist())}

# Drug name to DrugBank ID
drugbank = pd.read_csv('data/raw/drugbank.tsv', sep='\t')
string_protein_ids = aliases_df['#string_protein_id']


# cell
cell_feats_df = pd.read_csv('data/cell_tpm.csv', index_col=0)
gene_list = cell_feats_df.keys().tolist()[1:]

if not os.path.exists('baselines/GraphSynergy-master/data_ours/gene_new.json'):
    GENE_TO_STRING_DICT = {}

    for gene in gene_list:
        string_protein_id = string_protein_ids[aliases_df['alias'] == gene]
        if len(string_protein_id) == 0:
            print(f'protein {gene} not found.')
            continue
        GENE_TO_STRING_DICT[gene] = string_protein_id.iloc[0]

    gene_dict = json.dumps(GENE_TO_STRING_DICT)
    with open('baselines/GraphSynergy-master/data_ours/gene_new.json', 'w+') as file:
        file.write(gene_dict)
else:
    with open('baselines/GraphSynergy-master/data_ours/gene_new.json', 'r+') as file:
        GENE_TO_STRING_DICT = file.read()
    GENE_TO_STRING_DICT = json.loads(GENE_TO_STRING_DICT)

gene_keys = list(GENE_TO_STRING_DICT.keys())

threshold = 400
# pdb.set_trace()
cell_protein_df = []
# pdb.set_trace()
pairs = np.argwhere(cell_feats_df[gene_keys].to_numpy() > threshold)


# cell_names = list(map(lambda x: cell_feats_df['cell_line_names'][x], pairs[:, 0].tolist()))

cell_proteins = []
for i in range(len(pairs)):
    if GENE_TO_STRING_DICT[gene_keys[pairs[i, 1]]] in ENSP_TO_NODE_DICT.keys():
        cell_proteins.append([
            cell_feats_df['cell_line_names'][pairs[i, 0]],
            ENSP_TO_NODE_DICT[GENE_TO_STRING_DICT[gene_keys[pairs[i, 1]]]]
            ])


cell_protein_df = pd.DataFrame(cell_proteins, columns=['cell', 'protein'])
savedir = 'baselines/GraphSynergy-master/data_ours/'
os.makedirs(savedir, exist_ok=True)
cell_protein_df.to_csv(os.path.join(savedir, 'cell_protein.csv'), index=False)
# pdb.set_trace()














# protein1s = ppis_df['protein1']
# protein2s = ppis_df['protein2']

# proteins = set()

# for protein1, protein2 in zip(protein1s, protein2s):
#     proteins.add(protein1)
#     proteins.add(protein2)

# print(f'protine num: {len(proteins)}')

# proteins = list(sorted(proteins))

# PROTEIN_TO_ID_DICT = {}

# for pr in proteins:
#     gene_id = gene_entrez_ids[gene_names == pr]
#     if len(gene_id) == 0:
#         # print(f'protein {pr} cannot be found. Try to find it in the alias file.')
#         string_protein_id = string_protein_ids[aliases_df['alias'] == pr]
#         if len(string_protein_id) == 0:
#             print(f'protein {pr} still not be found.')
#             continue
#         # pdb.set_trace()
#         aliases_curr = aliases_df[string_protein_ids == string_protein_id.iloc[0]]
#         entrez_sources = ['Ensembl_HGNC_Entrez_Gene_ID', 'Ensembl_UniProt_DR_GeneID', 'BLAST_KEGG_GENEID']
#         for entrez_source in entrez_sources:
#             gene_id = aliases_curr['alias'][aliases_curr['source'] == entrez_source]
#             if len(gene_id) > 0:
#                 break
#         if len(gene_id) == 0:
#             print(f'protein {pr} can not be found.')
#             continue
#         # pdb.set_trace()
#     PROTEIN_TO_ID_DICT[pr] = int(gene_id.iloc[0])

# protein1_ids = []
# protein2_ids = []

# for protein1, protein2 in zip(protein1s, protein2s):
#     protein1_ids.append(PROTEIN_TO_ID_DICT[protein1])
#     protein2_ids.append(PROTEIN_TO_ID_DICT[protein2])
#     pdb.set_trace()