import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import pdb

protein_seq_df = pd.read_csv('data/protein_seq.csv')

protein_seq_proc = []

for i in tqdm(range(len(protein_seq_df))):
    seq_curr = protein_seq_df['Sequences'].iloc[i]
    # pdb.set_trace()
    protein_seq_proc.append(" ".join([s for s in seq_curr]))
    # pdb.set_trace()


pretein_seq_df_new = pd.DataFrame.from_dict(
    {
        "Proteins": protein_seq_df['Proteins'].tolist(),
        "Sequences": protein_seq_proc
    }
)

pretein_seq_df_new.to_csv('data/protein_seq_processed.csv', index=False)
