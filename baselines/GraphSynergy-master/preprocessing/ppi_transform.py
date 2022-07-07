import numpy as np
import pandas as pd
import pdb

COO_FILE = 'baselines/PRODeepSyn-main/cell/data_ours/ppi.coo.npy'
ppi_edges = np.load(COO_FILE).astype(int).transpose()

# pdb.set_trace()
protein_a = ppi_edges[0, :].tolist()
protein_b = ppi_edges[1, :].tolist()


ppi_df = pd.DataFrame.from_dict(
    {
        'protein_a': protein_a,
        'protein_b': protein_b}
)

ppi_df.to_excel('baselines/GraphSynergy-master/data_ours/protein-protein_network.xlsx', sheet_name='Human Interactome', index=False)