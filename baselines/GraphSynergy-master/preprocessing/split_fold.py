import pandas as pd
import numpy as np
import pdb
import os

fold_num = 3

DATANAME=f'data_ours_{fold_num}fold'

data_ori_dir = f'baselines/DeepDDs-master/{DATANAME}/'

for i in range(fold_num):
    savedir = f'baselines/GraphSynergy-master/{DATANAME}/fold{i}'

    dataset_train = os.path.join(data_ori_dir, f'train_fold{i}.csv')
    dataset_test = os.path.join(data_ori_dir, f'test_fold{i}.csv')

    dds_train_df = pd.read_csv(dataset_train)
    dds_test_df = pd.read_csv(dataset_test)

    for dds_df, type in zip([dds_train_df, dds_test_df], ['train', 'test']):
        cell_list = dds_df['cell'].tolist()
        drug1_db_list = dds_df['drug1_name'].tolist()
        drug2_db_list = dds_df['drug2_name'].tolist()
        synergy_list = dds_df['label'].tolist()

        dds_df_new = pd.DataFrame.from_dict(
            {
                'cell': cell_list,
                'drug1_db': drug1_db_list,
                'drug2_db': drug2_db_list,
                'synergy': synergy_list}
        )

        os.makedirs(os.path.join(savedir, type), exist_ok=True)
        dds_df_new.to_csv(os.path.join(savedir, type, 'drug_combinations.csv'), index=False)
