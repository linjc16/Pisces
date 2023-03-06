import pandas as pd
import os
import pdb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--fold', type=str)
args = parser.parse_args()

# RAW_DATA_DIR = f'/data/linjc/dds/data/leave_cells/fold{args.fold}'
# OUTPUT_DIR = 'baselines/GraphSynergy-master/data_ours_leave_cell'

# RAW_DATA_DIR = f'/data/linjc/dds/data/leave_combs/fold{args.fold}'
# OUTPUT_DIR = 'baselines/GraphSynergy-master/data_ours_leave_comb'

# RAW_DATA_DIR = f'/home/linjc/pisces/Pisces/leave_combs/fold{args.fold}'
# OUTPUT_DIR = 'baselines/GraphSynergy-master/data_ours_leave_comb_new'

RAW_DATA_DIR = f'/home/linjc/pisces/Pisces/transductive/fold{args.fold}'
OUTPUT_DIR = 'baselines/GraphSynergy-master/data_ours_new'


output_dir = os.path.join(OUTPUT_DIR, f'fold{args.fold}')
os.makedirs(output_dir, exist_ok=True)

train_savedir = os.path.join(output_dir, 'train')
os.makedirs(train_savedir, exist_ok=True)

test_savedir = os.path.join(output_dir, 'test')
os.makedirs(test_savedir, exist_ok=True)

valid_savedir = os.path.join(output_dir, 'valid')
os.makedirs(valid_savedir, exist_ok=True)

raw_data_train_dir = os.path.join(RAW_DATA_DIR, 'train.csv')
raw_data_valid_dir = os.path.join(RAW_DATA_DIR, 'valid.csv')
raw_data_test_dir = os.path.join(RAW_DATA_DIR, 'test.csv')

raw_data_train = pd.read_csv(raw_data_train_dir)
raw_data_test = pd.read_csv(raw_data_test_dir)
raw_data_valid = pd.read_csv(raw_data_valid_dir)

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

data_valid_df = pd.DataFrame.from_dict(
    {
        'cell': raw_data_valid['cell_line_names'].tolist(),
        'drug1_db': raw_data_valid['anchor_names'].tolist(),
        'drug2_db': raw_data_valid['library_names'].tolist(),
        'synergy': raw_data_valid['labels'].tolist()
    }
)

data_valid_df.to_csv(os.path.join(valid_savedir, 'drug_combinations.csv'), index=False)