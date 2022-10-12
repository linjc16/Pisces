import numpy as np
import pdb
import torch
import pandas as pd
import os
from tqdm import tqdm
from sklearn import metrics as sk_metrics


num_fold = 5
model_type = 'graph' # smiles


def generate_combs_ids():
    drug_combs = set()
    for fold in range(num_fold):
        test_data_path = f'/data/linjc/dds/data_new/transductive/fold{fold}/test.csv'
        test_data = pd.read_csv(test_data_path)
        for i in tqdm(range(len(test_data))):
            data_curr = test_data.iloc[i, :]
            drug_combs.add((data_curr['anchor_names'], data_curr['library_names']))
    
    drug_combs = list(drug_combs)
    drug_combs_df = []
    for i in tqdm(range(len(drug_combs))):
        drug_combs_df.append([i, drug_combs[i][0], drug_combs[i][1]])
    
    drug_combs_df = pd.DataFrame(drug_combs_df, columns=['id', 'anchor_names', 'library_names'])
    drug_combs_df.to_csv('data/drug_combs_ids.csv', index=False)

# generate_combs_ids()
# pdb.set_trace()

root_dir = 'dds/scripts/visualization/raw_output_scatter'

fold = 0


cell_types = []
predic = []
y_test = []
ids = []
test_data = []

for fold in range(num_fold):
    cell_types_path = os.path.join(root_dir, f'cell_types_trans_{model_type}_fold{fold}.npy')
    ids_path = os.path.join(root_dir, f'ids_trans_{model_type}_fold{fold}.pt')
    predic_path = os.path.join(root_dir, f'predic_trans_{model_type}_fold{fold}.npy')
    y_test_path = os.path.join(root_dir, f'y_test_trans_{model_type}_fold{fold}.npy')
    test_data_path = f'/data/linjc/dds/data_new/transductive/fold{fold}/test.csv'

    cell_types_curr = np.load(cell_types_path)
    predic_curr = np.load(predic_path)
    y_test_curr = np.load(y_test_path)
    ids_curr = torch.load(ids_path).numpy()

    test_data_curr = pd.read_csv(test_data_path)
    test_data_curr = test_data_curr.iloc[ids_curr, :]

    cell_types.append(cell_types_curr)
    predic.append(predic_curr)
    y_test.append(y_test_curr)
    ids.append(ids_curr)
    test_data.append(test_data_curr)

cell_types = np.concatenate(cell_types)
predic = np.concatenate(predic)
y_test = np.concatenate(y_test)
ids = np.concatenate(ids)
test_data = pd.concat(test_data)


drug_combs = pd.read_csv('data/drug_combs_ids.csv')

COMBS_TO_INDEX_DICT = {(drug_combs.iloc[idx, :]['anchor_names'], drug_combs.iloc[idx, :]['library_names']): idx for idx in range(len(drug_combs))}

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
drug_combs_results_df.to_csv(f'comb_level_trans_results_{model_type}.csv')


pdb.set_trace()

