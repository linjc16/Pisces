
import pandas as pd
from collections import defaultdict
import pdb
from tqdm import tqdm

df_all_data = pd.read_csv('data_feat/rnaseq_all_data_20220510.csv')


cell_lines = pd.read_csv('data/cell_features.csv')['cell_line_names']

df_cell_feats = defaultdict(list)

pdb.set_trace()

for i in tqdm(range(len(cell_lines))):
    raw_feats = df_all_data[df_all_data['model_name'] == cell_lines.iloc[i]]
    feat_names = raw_feats['gene_symbol']
    if len(feat_names) == 0:
        print(f'{cell_lines.iloc[i]} not found')
        continue
    # print(len(feat_names))
    if len(feat_names) != 37263:
        print(f'{cell_lines.iloc[i]} related genes num not match')
        continue
    df_cell_feats['cell_line_names'].append(cell_lines.iloc[i])
    len_curr = len(df_cell_feats['cell_line_names'])
    for j in range(len(feat_names)):
        if feat_names.iloc[j] == 'SEPTIN4':
            continue
        df_cell_feats[feat_names.iloc[j]].append(raw_feats['tpm'].iloc[j])

for key in df_cell_feats.keys():
    if len(df_cell_feats[key]) != 125:
        print(f'{key} not equal to 125.')
pdb.set_trace()
df_cell_feats = pd.DataFrame.from_dict(df_cell_feats)
df_cell_feats.to_csv('data/cell_tpm.csv')