import numpy as np
import os
import pdb
import torch
import pandas as pd
from tqdm import tqdm
from sklearn import metrics as sk_metrics

root_dir = 'dds/scripts/case_study/raw_output'

cell_types = np.load(os.path.join(root_dir, 'cell_types_leave_combs.npy'))
predic = np.load(os.path.join(root_dir, 'predic_leave_combs.npy'))
y_test = np.load(os.path.join(root_dir, 'y_test_leave_combs.npy'))
pred = (predic >= 0.5).astype(np.int64)
ids = torch.load(os.path.join(root_dir, 'ids_leave_combs.pt')).numpy()


test_data = pd.read_csv("/data/linjc/dds/data_new/leave_combs/fold0/test.csv")
test_data = test_data.iloc[ids, :]

# pdb.set_trace()
# drug_combs = set()

# for i in tqdm(range(len(test_data))):
#     data_curr = test_data.iloc[i, :]
#     drug_combs.add((data_curr['anchor_names'], data_curr['library_names']))

# drug_combs = list(drug_combs)

# COMBS_TO_INDEX_DICT = {item: idx for idx, item in enumerate(drug_combs)}

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
    pred_curr = (predic_curr >= 0.5).astype(np.int64)
    # pdb.set_trace()
    bacc_curr = sk_metrics.balanced_accuracy_score(y_test_curr, pred_curr)
    kappa_curr = sk_metrics.cohen_kappa_score(y_test_curr, pred_curr)
    f1_score_curr = sk_metrics.f1_score(y_test_curr, pred_curr)
    p, r, t = sk_metrics.precision_recall_curve(y_test_curr, predic_curr)
    auc_prc_curr = sk_metrics.auc(r, p)


    drug_combs_results_df.append([i, drug_combs.iloc[i, :]['anchor_names'], drug_combs.iloc[i, :]['library_names'], auc_prc_curr, f1_score_curr, bacc_curr, kappa_curr, y_test_curr.sum(), len(y_test_curr)])

drug_combs_results_df = pd.DataFrame(drug_combs_results_df, columns=['id', 'anchor_names', 'library_names', 'AUPRC', 'F1', 'BACC', 'KAPPA', 'num_pos', 'data_num'])
drug_combs_results_df.to_csv('comb_level_results.csv')

# drug_combs_df = []

# for i in tqdm(range(len(drug_combs))):
#     drug_combs_df.append([i, drug_combs[i][0], drug_combs[i][1]])

# drug_combs_df = pd.DataFrame(drug_combs_df, columns=['id', 'anchor_names', 'library_names'])
# drug_combs_df.to_csv('data/drug_combs_ids.csv', index=False)

# cell_df = pd.read_csv('data/cell_tpm.csv')

# cell_names = cell_df['cell_line_names'].tolist()

# cell_new_df = pd.DataFrame.from_dict(
#     {
#         'cell_line_names': cell_names,
#         'index': list(range(len(cell_names)))
#     }
# )

# cell_new_df.to_csv('data/cell_ids.csv', index=False)

# pdb.set_trace()