from cgi import test
import pandas as pd
import os
import pdb
from collections import defaultdict
from requests import head

# df_dds = pd.read_csv('baselines/PRODeepSyn-main/predictor/data/synergy.tsv', sep='\t')

fold_files = [f'baselines/DeepDDs-master/data_ours/test_fold{i}.csv' for i in range(5)]

df_synergy = defaultdict(list)

for index, fold_file in enumerate(fold_files):
    test_fold = pd.read_csv(fold_file)
    df_synergy['drug_a'].extend(test_fold['drug1_name'].tolist())
    df_synergy['drug_b'].extend(test_fold['drug2_name'].tolist())
    df_synergy['cell_line'].extend(test_fold['cell'].tolist())
    df_synergy['label'].extend(test_fold['label'].tolist())
    df_synergy['fold'].extend([index] * len(test_fold))

    
df_synergy = pd.DataFrame.from_dict(df_synergy)
df_synergy.to_csv('baselines/PRODeepSyn-main/predictor/data_ours/synergy.tsv', sep='\t', index=False, header=True)

pdb.set_trace()
