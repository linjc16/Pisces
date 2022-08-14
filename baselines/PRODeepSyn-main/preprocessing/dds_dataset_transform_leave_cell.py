import pandas as pd
import numpy as np
import pdb
import os
from tqdm import tqdm
from collections import defaultdict



num_fold = 3

for i in range(num_fold):
    df_synergy = defaultdict(list)
    data_train = pd.read_csv(f'baselines/DeepDDs-master/data_ours_leave_cell/train_fold{i}.csv')
    data_test = pd.read_csv(f'baselines/DeepDDs-master/data_ours_leave_cell/test_fold{i}.csv')

    df_synergy['drug_a'].extend(data_test['drug1_name'].tolist())
    df_synergy['drug_b'].extend(data_test['drug2_name'].tolist())
    df_synergy['cell_line'].extend(data_test['cell'].tolist())
    df_synergy['label'].extend(data_test['label'].tolist())
    df_synergy['fold'].extend([0] * len(data_test))

    df_synergy['drug_a'].extend(data_train['drug1_name'].tolist())
    df_synergy['drug_b'].extend(data_train['drug2_name'].tolist())
    df_synergy['cell_line'].extend(data_train['cell'].tolist())
    df_synergy['label'].extend(data_train['label'].tolist())
    df_synergy['fold'].extend([1] * len(data_train))

    df_synergy = pd.DataFrame.from_dict(df_synergy)

    df_synergy.to_csv(f'baselines/PRODeepSyn-main/predictor/data_ours_leave_cell/synergy_fold{i}.tsv', sep='\t', index=False, header=True)
