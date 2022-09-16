import pandas as pd
import pdb
import os
import numpy as np

fold_num = 5
RAW_DATA_DIR = 'baselines/DeepDDs-master/data_ours'
OUTPUT_DIR = '/data/linjc/dds/data_new/transductive'

for i in range(fold_num):

    output_dir = os.path.join(OUTPUT_DIR, f'fold{i}')
    os.makedirs(output_dir, exist_ok=True)

    raw_data_train_dir = os.path.join(RAW_DATA_DIR, f'train_fold{i}.csv')
    raw_data_test_dir = os.path.join(RAW_DATA_DIR, f'test_fold{i}.csv')

    raw_data_train = pd.read_csv(raw_data_train_dir)
    raw_data_test = pd.read_csv(raw_data_test_dir)

    data_train_df = pd.DataFrame.from_dict(
        {
            'cell_line_names': raw_data_train['cell'].tolist(),
            'anchor_names': raw_data_train['drug1_name'].tolist(),
            'library_names': raw_data_train['drug2_name'].tolist(),
            'labels': raw_data_train['label'].tolist()
        }
    )
    data_train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)

    data_test_df = pd.DataFrame.from_dict(
        {
            'cell_line_names': raw_data_test['cell'].tolist(),
            'anchor_names': raw_data_test['drug1_name'].tolist(),
            'library_names': raw_data_test['drug2_name'].tolist(),
            'labels': raw_data_test['label'].tolist()
        }
    )
    
    data_test_df.to_csv(os.path.join(output_dir, 'valid.csv'), index=False)
    data_test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    
    # pdb.set_trace()
