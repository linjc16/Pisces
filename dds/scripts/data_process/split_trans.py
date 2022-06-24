from pickle import TRUE
import pandas as pd
import pdb
import os
import numpy as np
df_ddses = pd.read_csv('data/ddses.csv', index_col=0)

output_dir = 'data/transductive/fold1'

os.makedirs(output_dir, exist_ok=True)

df_ddses = df_ddses.iloc[np.random.permutation(len(df_ddses))]
# df_train_pos = df_ddses[df_ddses['labels'] == 1]
# df_train_neg = df_ddses[df_ddses['labels'] == 0]

# pdb.set_trace()

df_ddses[0:90000].to_csv(os.path.join(output_dir, 'train.csv'))
df_ddses[90000:105000].to_csv(os.path.join(output_dir, 'valid.csv'))
df_ddses[105000:].to_csv(os.path.join(output_dir, 'test.csv'))

pdb.set_trace()
