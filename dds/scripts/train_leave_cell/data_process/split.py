import pandas as pd
import pdb
import os
import numpy as np
import random
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=str)
parser.add_argument('--fold', type=str)
args = parser.parse_args()

RAW_DATA_DIR = 'data/ddses_new.csv'
OUTPUT_DIR = '/data/linjc/dds/data/leave_cells'
CELL_DIR = 'data/cell_tpm.csv'

output_dir = os.path.join(OUTPUT_DIR, f'fold{args.fold}')
os.makedirs(output_dir, exist_ok=True)

raw_data_df = pd.read_csv(RAW_DATA_DIR)
cell_data_df = pd.read_csv(CELL_DIR)

cell_list = cell_data_df['cell_line_names'].tolist()

random.seed(args.seed)
random.shuffle(cell_list)

cell_train = set(cell_list[:100])
cell_test = set(cell_list[100:])

data_train_df = []
data_test_df = []

for i in tqdm(range(len(raw_data_df))):
    data_curr = raw_data_df.iloc[i, :]
    cell_curr = data_curr['cell_line_names']
    if cell_curr in cell_train:
        data_train_df.append(data_curr.tolist())
    else:
        data_test_df.append(data_curr.tolist())

print(f'num training sample: {len(data_train_df)}')
print(f'num test sample: {len(data_test_df)}')
data_train_df = pd.DataFrame(data_train_df, columns=['cell_line_names', 'anchor_names', 'library_names', 'labels'])
data_train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)

data_test_df = pd.DataFrame(data_test_df, columns=['cell_line_names', 'anchor_names', 'library_names', 'labels'])
data_test_df.to_csv(os.path.join(output_dir, 'valid.csv'), index=False)
data_test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)

