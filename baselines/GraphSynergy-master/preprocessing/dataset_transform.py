import pandas as pd
import pdb
from tqdm import tqdm

# convert drug-target files

# ppis_df = pd.read_csv('data/ppi.csv')
drug_target_ori = pd.read_csv('data/raw/41467_2019_9186_MOESM4_ESM.csv')
gene_info = pd.read_csv('data/raw/gene_identifiers_20191101.csv')
aliases_df = pd.read_csv('data/raw/9606.protein.aliases.v11.5.txt', sep='\t', names=['#string_protein_id', 'alias', 'source'], header=0)

drug_smiles_df = pd.read_csv('data/drug_smiles.csv', index_col=0)

# Drug name to DrugBank ID
drugbank = pd.read_csv('data/raw/drugbank.tsv', sep='\t')

drug_smiles_df_new = []

# for drug_name in drug_smiles_df['drug_names']:
#     db_id = drugbank['drugbank_id'][drugbank['name'] == drug_name]
#     if len(db_id) == 0:
#         print(f'drug {drug_name} not found.')
#         continue
#     drug_smiles_df_new.append([drug_name, db_id])

# drug_smiles_df_new = pd.DataFrame(drug_smiles_df_new, columns=['drug name', 'drugbank_id'])
# pdb.set_trace()


# pdb.set_trace()

string_protein_ids = aliases_df['#string_protein_id']


gene_names = gene_info['cosmic_gene_symbol']
gene_entrez_ids = gene_info['entrez_id']


# cell
cell_feats_df = pd.read_csv('data/cell_feats.csv', index_col=0)

gene_list = cell_feats_df.keys().tolist()[1:]

GENE_TO_STRING_DICT = {}
# pdb.set_trace()

for gene in gene_list:
    string_protein_id = string_protein_ids[aliases_df['alias'] == gene]
    if len(string_protein_id) == 0:
        print(f'protein {gene} not found.')
        continue
    GENE_TO_STRING_DICT[gene] = string_protein_id.iloc[0]


gene_df_ours = pd.DataFrame.from_dict(GENE_TO_STRING_DICT)
gene_df_ours.to_csv('baselines/GraphSynergy-master/data_ours/gene_new.csv')

threshold = 20
# pdb.set_trace()
cell_protein_df = []

for i in tqdm(range(len(cell_feats_df))):
    for key in GENE_TO_STRING_DICT.keys():
        if cell_feats_df.iloc[0, 1:][key] > threshold:
            cell_name = cell_feats_df.iloc[i, 0]
            
            cell_protein_df.append([cell_name, GENE_TO_STRING_DICT[key]])

cell_protein_df = pd.DataFrame(cell_protein_df, columns=['cell_line_names', 'proteins'])
cell_protein_df.to_csv('baselines/GraphSynergy-master/data_ours/cell_protein.csv')
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