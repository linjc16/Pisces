import numpy as np
import pdb
import os
import torch
import pandas as pd
from tqdm import tqdm
from sklearn import metrics as sk_metrics

root_dir = 'dds/scripts/case_study/raw_output'

data_processed_df = pd.read_csv('baselines/GraphSynergy-master/data_ours_leave_comb/fold0/test/drug_combination_processed.csv')
data_raw_df = pd.read_csv('baselines/GraphSynergy-master/data_ours_leave_comb/fold0/test/drug_combinations.csv')

INDEX_TO_CELL_DICT = {}
INDEX_TO_DRUG_DICT = {}

for i in tqdm(range(len(data_processed_df))):
    data_curr_processed = data_processed_df.iloc[i, :]
    data_curr_raw = data_raw_df.iloc[i, :]

    if data_curr_processed['cell'] not in INDEX_TO_CELL_DICT.keys():
        INDEX_TO_CELL_DICT[data_curr_processed['cell']] = data_curr_raw['cell']
    
    if data_curr_processed['drug1_db'] not in INDEX_TO_DRUG_DICT.keys():
        INDEX_TO_DRUG_DICT[data_curr_processed['drug1_db']] = data_curr_raw['drug1_db']

    if data_curr_processed['drug2_db'] not in INDEX_TO_DRUG_DICT.keys():
        INDEX_TO_DRUG_DICT[data_curr_processed['drug2_db']] = data_curr_raw['drug2_db']

INDEX_TO_CELL_DICT = dict(sorted(INDEX_TO_CELL_DICT.items(),key=lambda x:x[0],reverse=False))
INDEX_TO_DRUG_DICT = dict(sorted(INDEX_TO_DRUG_DICT.items(),key=lambda x:x[0],reverse=False))


predic = torch.load(os.path.join(root_dir, 'y_pred_GS_all.pt')).numpy()
data = torch.load(os.path.join(root_dir, 'data_GS_all.pt')).numpy()

# pdb.set_trace()
y_test = data[:, 3]
pred = (predic >= 0.5).astype(np.int64)




drug_combs = pd.read_csv('data/drug_combs_ids.csv')

COMBS_TO_INDEX_DICT = {(drug_combs.iloc[idx, :]['anchor_names'], drug_combs.iloc[idx, :]['library_names']): idx for idx in range(len(drug_combs))}

# pdb.set_trace()

comb_types = []

for i in tqdm(range(len(data))):
    drugA, drugB = INDEX_TO_DRUG_DICT[data[i, 1]], INDEX_TO_DRUG_DICT[data[i, 2]]
    comb_types.append(COMBS_TO_INDEX_DICT[(drugA, drugB)])    

# pdb.set_trace()

comb_types = np.array(comb_types)

drug_combs_results_df = []



for i in range(len(drug_combs)):
    predic_curr = predic[comb_types == i]
    y_test_curr = y_test[comb_types == i]
    pred_curr = pred[comb_types == i]
    # pdb.set_trace()
    bacc_curr = sk_metrics.balanced_accuracy_score(y_test_curr, pred_curr)
    kappa_curr = sk_metrics.cohen_kappa_score(y_test_curr, pred_curr)
    f1_score_curr = sk_metrics.f1_score(y_test_curr, pred_curr)
    p, r, t = sk_metrics.precision_recall_curve(y_test_curr, predic_curr)
    auc_prc_curr = sk_metrics.auc(r, p)

    drug_combs_results_df.append([i, drug_combs.iloc[i, :]['anchor_names'], drug_combs.iloc[i, :]['library_names'], auc_prc_curr, f1_score_curr, bacc_curr, kappa_curr, y_test_curr.sum(), len(y_test_curr)])

drug_combs_results_df = pd.DataFrame(drug_combs_results_df, columns=['id', 'anchor_names', 'library_names', 'AUPRC', 'F1', 'BACC', 'KAPPA', 'num_pos', 'data_num'])
drug_combs_results_df.to_csv('comb_level_results_graphSynergy.csv')


# pdb.set_trace()