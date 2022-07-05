import os
import pandas as pd
import pdb
import numpy as np
from tqdm import tqdm

savedir = 'baselines/PRODeepSyn-main/cell/data_ours'

df_cell_feats = pd.read_csv('data/cell_read_count.csv', index_col=0)
cell_names = df_cell_feats['cell_line_names']

df_cell2id = {
    'cell': cell_names,
    'id': list(range(len(cell_names)))
}

df_cell2id = pd.DataFrame.from_dict(df_cell2id)

df_cell2id.to_csv(os.path.join(savedir, 'cell2id.tsv'), sep='\t', index=False, header=True)