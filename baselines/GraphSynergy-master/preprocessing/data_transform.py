import pandas as pd
import os
import pdb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--fold', type=str)
args = parser.parse_args()

RAW_DATA_DIR = f'/data/linjc/dds/data/leave_cells/fold{args.fold}'
OUTPUT_DIR = 'baselines/GraphSynergy-master/data_ours_leave_cell'

output_dir = os.path.join(OUTPUT_DIR, f'fold{args.fold}')
os.makedirs(output_dir, exist_ok=True)

train_savedir = os.path.join(output_dir, 'train')
os.makedirs(train_savedir, exist_ok=True)

test_savedir = os.path.join(output_dir, 'test')
os.makedirs(test_savedir, exist_ok=True)

raw_data_train_dir = os.path.join(RAW_DATA_DIR, f'train.csv')
raw_data_test_dir = os.path.join(RAW_DATA_DIR, f'test.csv')

raw_data_train = pd.read_csv(raw_data_train_dir)
raw_data_test = pd.read_csv(raw_data_test_dir)

data_train_df = pd.DataFrame.from_dict(
    {
        'cell': raw_data_train['cell_line_names'].tolist(),
        'drug1_db': raw_data_train['anchor_names'].tolist(),
        'drug2_db': raw_data_train['library_names'].tolist(),
        'synergy': raw_data_train['labels'].tolist()
    }
)
data_train_df.to_csv(os.path.join(train_savedir, 'drug_combinations.csv'), index=False)

data_test_df = pd.DataFrame.from_dict(
    {
        'cell': raw_data_test['cell_line_names'].tolist(),
        'drug1_db': raw_data_test['anchor_names'].tolist(),
        'drug2_db': raw_data_test['library_names'].tolist(),
        'synergy': raw_data_test['labels'].tolist()
    }
)

data_test_df.to_csv(os.path.join(test_savedir, 'drug_combinations.csv'), index=False)
    