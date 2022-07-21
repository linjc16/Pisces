from email.contentmanager import raw_data_manager
import pandas as pd
import pdb
import os
import numpy as np

raw_data_dir = 'baselines/DeepDDs-master/data_ours/ddses_new.csv'
output_dir = 'data'

raw_data_df = pd.read_csv(raw_data_dir)

data_df = pd.DataFrame.from_dict(
    {
        'cell_line_names': raw_data_df['cell'].tolist(),
        'anchor_names': raw_data_df['drug1_name'].tolist(),
        'library_names': raw_data_df['drug2_name'].tolist(),
        'labels': raw_data_df['label'].tolist()
    }
)

data_df.to_csv(os.path.join(output_dir, 'ddses_new.csv'), index=False)