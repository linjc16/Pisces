import argparse
import os
import time
import torch
import torch.nn as nn
import pickle
import logging
from tqdm import tqdm
import glob

from datetime import datetime

from model.datasets import FastSynergyDataset, FastTensorDataLoader
from model.classifier import DNN
from model.utils import random_split_indices
from const import SYNERGY_FILE, DRUG2ID_FILE, DRUG_FEAT_FILE, CELL_FEAT_FILE, CELL2ID_FILE, OUTPUT_DIR

from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.metrics import cohen_kappa_score, accuracy_score, roc_auc_score, precision_score, recall_score, balanced_accuracy_score, f1_score
from sklearn import metrics
import pdb

time_str = str(datetime.now().strftime('%y%m%d%H%M'))

def step_batch_eval(model, batch, gpu_id=None):
    drug1_feats, drug2_feats, cell_feats, y_true = batch
    if gpu_id is not None:
        drug1_feats, drug2_feats, cell_feats, y_true = drug1_feats.cuda(gpu_id), drug2_feats.cuda(gpu_id), \
                                                       cell_feats.cuda(gpu_id), y_true.cuda(gpu_id)
    yp1 = model(drug1_feats, drug2_feats, cell_feats)
    yp2 = model(drug2_feats, drug1_feats, cell_feats)
    y_pred = (yp1 + yp2) / 2
    
    return y_true, y_pred, torch.sigmoid(y_pred) > 0.25

def test_epoch(model, loader, gpu_id=None):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_prelabels = torch.Tensor()
    with torch.no_grad():
        for batch in loader:
            targets, pred_scores, pred_labels  = step_batch_eval(model, batch, gpu_id)
            total_preds = torch.cat([total_preds, pred_scores.cpu()], dim=0)
            total_prelabels = torch.cat([total_prelabels, pred_labels.cpu()], dim=0)
            total_labels = torch.cat([total_labels, targets.cpu()], dim=0)

    return total_labels.numpy().flatten(), total_preds.numpy().flatten(), total_prelabels.numpy().flatten()


def eval_model(model, train_data, test_data,
               batch_size, gpu_id, mdl_dir):
    tr_indices, es_indices = random_split_indices(len(train_data), test_rate=0.1)
    train_loader = FastTensorDataLoader(*train_data.tensor_samples(tr_indices), batch_size=batch_size, shuffle=True)
    valid_loader = FastTensorDataLoader(*train_data.tensor_samples(es_indices), batch_size=len(es_indices) // 4)
    test_loader = FastTensorDataLoader(*test_data.tensor_samples(), batch_size=len(test_data) // 4)

    T, S, Y = test_epoch(model, test_loader, gpu_id)
    # pdb.set_trace()
    AUC = roc_auc_score(T, S)
    precision, recall, threshold = metrics.precision_recall_curve(T, S)
    PR_AUC = metrics.auc(recall, precision)
    BACC = balanced_accuracy_score(T, Y)
    tn, fp, fn, tp = confusion_matrix(T, Y).ravel()
    TPR = tp / (tp + fn)
    PREC = precision_score(T, Y)
    ACC = accuracy_score(T, Y)
    KAPPA = cohen_kappa_score(T, Y)
    recall = recall_score(T, Y)
    F1 = f1_score(T, Y)
    logging.info(f'ACC: {ACC}, BACC: {BACC}, AUC: {AUC}, PR_AUC: {PR_AUC}, \
            PREC: {PREC}, RECALL: {recall}, F1: {F1}, TPR: {TPR}, KAPPA: {KAPPA}.')


def create_model(data, hidden_size, gpu_id=None):
    model = DNN(data.cell_feat_len() + 2 * data.drug_feat_len(), hidden_size)
    if gpu_id is not None:
        model = model.cuda(gpu_id)
    return model

def cv(args, out_dir):

    if torch.cuda.is_available() and (args.gpu is not None):
        gpu_id = args.gpu
    else:
        gpu_id = None

    n_folds = 5
    n_delimiter = 60
    best_hs = args.best_hs

    for test_fold in range(n_folds):
        outer_trn_folds = [x for x in range(n_folds) if x != test_fold]
        logging.info("Outer: train folds {}, test folds {}".format(outer_trn_folds, test_fold))
        logging.info("-" * n_delimiter)
        param = []
        losses = []

        train_data = FastSynergyDataset(DRUG2ID_FILE, CELL2ID_FILE, DRUG_FEAT_FILE, CELL_FEAT_FILE,
                                        SYNERGY_FILE, use_folds=outer_trn_folds)
        test_data = FastSynergyDataset(DRUG2ID_FILE, CELL2ID_FILE, DRUG_FEAT_FILE, CELL_FEAT_FILE,
                                       SYNERGY_FILE, use_folds=[test_fold], train=False)
        model = create_model(train_data, best_hs, gpu_id)
        test_mdl_dir = os.path.join(out_dir, str(test_fold))

        ckpt_path = glob.glob(os.path.join(test_mdl_dir, '*.pkl'))

        model.load_state_dict(torch.load(ckpt_path[-1]))

        logging.info("Best hidden size: {}".format(best_hs))
        logging.info("Start test on fold {}.".format([test_fold]))
        
        eval_model(model, train_data, test_data,
                               args.batch, gpu_id, test_mdl_dir)
        logging.info("*" * n_delimiter + '\n')
    logging.info("CV completed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=256, help="batch size")
    parser.add_argument('--gpu', type=int, default=None, help="cuda device")
    parser.add_argument('--suffix', type=str, default=time_str, help="model dir suffix")
    parser.add_argument('--best_hs', type=int, default=2048, help="best hidden size")
    args = parser.parse_args()
    
    out_dir = os.path.join(OUTPUT_DIR, 'cv_{}'.format(args.suffix))
    log_file = os.path.join(out_dir, 'inf.log')
    logging.basicConfig(filename=log_file,
                        format='%(asctime)s %(message)s',
                        datefmt='[%Y-%m-%d %H:%M:%S]',
                        level=logging.INFO)
    cv(args, out_dir)


if __name__ == '__main__':
    main()