import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import pdb


if __name__ == '__main__':
    fold_num = 5

    savedir = 'baselines/DeepDDs-master/data_ours'
    os.makedirs(savedir, exist_ok=True)

    RAW_DATA_DIR = '/home/linjc/pisces/Pisces/transductive/'
    for i in range(fold_num):
        DATA_DIR_FOLD = os.path.join(RAW_DATA_DIR, f'fold{i}')
        df_train_data = pd.read_csv(os.path.join(DATA_DIR_FOLD, 'train.csv'))
        df_valid_data = pd.read_csv(os.path.join(DATA_DIR_FOLD, 'valid.csv'))
        df_test_data = pd.read_csv(os.path.join(DATA_DIR_FOLD, 'test.csv'))

        df_drug_smiles = pd.read_csv('data/drug_smiles.csv', index_col=0)

        all_drug_dict = {}
        for idx, smiles in zip(tqdm(df_drug_smiles['drug_names']), df_drug_smiles['smiles']):
            all_drug_dict[idx] = smiles


        def convert(df_ddses, type, fold):
            dds_data = []
            # transform ddses [drug1_SMILES, drug2_SMILES, cell, label]
            for drug1, drug2, cell, label in zip(tqdm(df_ddses['anchor_names']), df_ddses['library_names'], \
                df_ddses['cell_line_names'], df_ddses['labels']):

                dds_data.append([drug1, drug2, cell, label, all_drug_dict[drug1], all_drug_dict[drug2]])
            

            new_data_df = pd.DataFrame(dds_data, columns=['drug1_name', 'drug2_name', 'cell', 'label', 'drug1', 'drug2'])
            new_data_df.to_csv(os.path.join(savedir, f'{type}_fold{fold}.csv'), index=False)

        convert(df_train_data, 'train', i)
        convert(df_valid_data, 'valid', i)
        convert(df_test_data, 'test', i)

