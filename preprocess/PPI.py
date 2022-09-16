import pandas as pd
import pdb


ppi_df = pd.read_csv('data/raw/9606.protein.links.v11.5.txt', sep=' ', names=['protein1', 'protein2', 'combined_score'], header=0)
p_info_df = pd.read_csv('data/raw/9606.protein.info.v11.5.txt', sep='\t', names=['#string_protein_id', 'preferred_name', 'protein_size', 'annotation'], header=0)

PROTEIN_TO_GENE_DICT = {protein: gene for protein, gene in zip(p_info_df['#string_protein_id'].tolist(), p_info_df['preferred_name'].tolist())}

p1_list = ppi_df['protein1']
p2_list = ppi_df['protein2']

g1_list = [PROTEIN_TO_GENE_DICT[p1] for p1 in p1_list]
g2_list = [PROTEIN_TO_GENE_DICT[p2] for p2 in p2_list]

ppi_df_new = pd.DataFrame.from_dict({'protein1': g1_list, 'protein2': g2_list})
ppi_df_new.to_csv('data/ppi.csv', index=False)


# pdb.set_trace()