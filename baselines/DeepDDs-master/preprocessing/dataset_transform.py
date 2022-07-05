import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import pdb


if __name__ == '__main__':
    savedir = 'baselines/DeepDDs-master/data_ours'
    os.makedirs(savedir, exist_ok=True)

    df_ddses = pd.read_csv('data/ddses.csv', index_col=0)
    cell_feats = pd.read_csv('data/cell_read_count.csv', index_col=0)
    df_drug_smiles = pd.read_csv('data/drug_smiles.csv', index_col=0)

    # pdb.set_trace()
    all_drug_dict = {}
    for idx, smiles in zip(tqdm(df_drug_smiles['drug_names']), df_drug_smiles['smiles']):
        all_drug_dict[idx] = smiles

    cell_lines = set(cell_feats['cell_line_names'])

    dds_data = []
    # transform ddses [drug1_SMILES, drug2_SMILES, cell, label]
    for drug1, drug2, cell, label in zip(tqdm(df_ddses['anchor_names']), df_ddses['library_names'], \
        df_ddses['cell_line_names'], df_ddses['labels']):

        if cell not in cell_lines:
            continue

        if drug1 in all_drug_dict.keys() and drug2 in all_drug_dict.keys():
            dds_data.append([drug1, drug2, cell, label, all_drug_dict[drug1], all_drug_dict[drug2]])
    

    new_data_df = pd.DataFrame(dds_data, columns=['drug1_name', 'drug2_name', 'cell', 'label', 'drug1', 'drug2'])
    new_data_df.to_csv(os.path.join(savedir, 'ddses_new.csv'))

    # transform cell lines [cell_line, features,...]
    # pdb.set_trace()
    # cell_feats_values = cell_feats.iloc[:, 1:]
    # cell_feats.iloc[:, 1:] = (cell_feats_values - cell_feats_values.mean()) / (cell_feats_values.std() + 1e-8)
    # pdb.set_trace()
    cell_feats.to_csv(os.path.join(savedir, 'cell_features_expression_read_count_new.csv'), index=False)
    
    # transform smiles
    df_drug_smiles_new = pd.DataFrame.from_dict({'drug_names': all_drug_dict.keys(), 'smile': all_drug_dict.values()})
    df_drug_smiles_new.to_csv(os.path.join(savedir, 'drug_smiles_new.csv'), index=False)

