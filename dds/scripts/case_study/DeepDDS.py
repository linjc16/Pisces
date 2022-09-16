import numpy as np
import pdb
import torch
import os
import pandas as pd
from tqdm import tqdm
from sklearn import metrics as sk_metrics

root_dir = 'dds/scripts/case_study/raw_output'

predic = np.load(os.path.join(root_dir, 'S_DeepDDS.npy'))
y_test = np.load(os.path.join(root_dir, 'T_DeepDDS.npy'))
pred = np.load(os.path.join(root_dir, 'Y_DeepDDS.npy'))


test_data = pd.read_csv("/data/linjc/dds/data_new/leave_combs/fold0/test.csv")


drug_combs = pd.read_csv('data/drug_combs_ids.csv')

COMBS_TO_INDEX_DICT = {(drug_combs.iloc[idx, :]['anchor_names'], drug_combs.iloc[idx, :]['library_names']): idx for idx in range(len(drug_combs))}

# pdb.set_trace()

comb_types = []

for i in tqdm(range(len(test_data))):
    data_curr = test_data.iloc[i, :]
    comb_types.append(COMBS_TO_INDEX_DICT[(data_curr['anchor_names'], data_curr['library_names'])])    

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
drug_combs_results_df.to_csv('comb_level_results_deepdds.csv')


# pdb.set_trace()