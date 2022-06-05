import pandas as pd
import pdb
import os
df_ddses = pd.read_csv('data/ddses.csv', index_col=0)

output_dir = 'data/transductive/fold1'


# df_train_pos = df_ddses[df_ddses['labels'] == 1]
# df_train_neg = df_ddses[df_ddses['labels'] == 0]

df_ddses[0:5000].to_csv(os.path.join(output_dir, 'train.csv'))
df_ddses[5000:7000].to_csv(os.path.join(output_dir, 'valid.csv'))
df_ddses[7000:9000].to_csv(os.path.join(output_dir, 'test.csv'))

# pdb.set_trace()
