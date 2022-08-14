from asyncio import FastChildWatcher
import pandas as pd
import os
from tqdm import tqdm
import pdb

savedir = 'data/protein_sequence'

protein_seq_df = pd.read_csv(os.path.join(savedir, 'protein_seq_processed.csv'))

protein_set = set(protein_seq_df['Proteins'].tolist())
tpm_df = pd.read_csv('data/cell_tpm.csv', index_col=0)

tpm_df_new = {}
tpm_df_new['cell_line_names'] = tpm_df['cell_line_names'].tolist()

tpm_cols = tpm_df.columns.tolist()

for i in tqdm(range(1, len(tpm_df.columns))):
    if tpm_cols[i] in protein_set:
        tpm_df_new[tpm_cols[i]] = tpm_df.iloc[:, i].tolist()


tpm_df_new = pd.DataFrame.from_dict(tpm_df_new)
tpm_df_new.to_csv(os.path.join(savedir, 'cell_tpm_processed.csv'), index=False)

