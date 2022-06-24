import argparse
import os
import torch
import torch.nn as nn
import numpy as np
import pickle

from datetime import datetime


from model.datasets import FastSynergyDataset, FastTensorDataLoader
from model.models import DNN
from model.utils import conf_inv, calc_stat, find_best_model
from const import SYNERGY_FILE, DRUG2ID_FILE, DRUG_FEAT_FILE, CELL_FEAT_FILE, CELL2ID_FILE, OUTPUT_DIR

time_str = str(datetime.now().strftime('%y%m%d%H%M'))


def create_model(data, hidden_size, gpu_id=None):
    model = DNN(data.cell_feat_len() + 2 * data.drug_feat_len(), hidden_size)
    if gpu_id is not None:
        model = model.cuda(gpu_id)
    return model


def calc_metrics(out_dir):
    n_folds = 5
    n_delimiter = 60
    loss_func = nn.MSELoss(reduction='sum')
    test_losses = []
    pearson_coefs = []
    y_preds = []
    y_trues = []
    for test_fold in range(n_folds):
        test_data = FastSynergyDataset(DRUG2ID_FILE, CELL2ID_FILE, DRUG_FEAT_FILE, CELL_FEAT_FILE,
                                       SYNERGY_FILE, use_folds=[test_fold], train=False)
        test_mdl_dir = os.path.join(out_dir, str(test_fold))
        try:
            model = create_model(test_data, 4096, None)
            model.load_state_dict(torch.load(find_best_model(test_mdl_dir), map_location=torch.device('cpu')))
        except Exception:
            try:
                model = create_model(test_data, 8192, None)
                model.load_state_dict(torch.load(find_best_model(test_mdl_dir), map_location=torch.device('cpu')))
            except Exception:
                model = create_model(test_data, 2048, None)
                model.load_state_dict(torch.load(find_best_model(test_mdl_dir), map_location=torch.device('cpu')))
        test_loader = FastTensorDataLoader(*test_data.tensor_samples(), batch_size=len(test_data))
        model.eval()
        with torch.no_grad():
            for drug1_feats, drug2_feats, cell_feats, y_true in test_loader:
                yp1 = model(drug1_feats, drug2_feats, cell_feats)
                yp2 = model(drug2_feats, drug1_feats, cell_feats)
                y_pred = (yp1 + yp2) / 2
                test_loss = loss_func(y_pred, y_true).item()
                y_pred = y_pred.numpy().flatten()
                y_true = y_true.numpy().flatten()
                y_preds.append(y_pred)
                y_trues.append(y_true)
                pc = np.corrcoef(y_pred, y_true)[0, 1]
                pearson_coefs.append(pc)
                test_loss /= len(y_true)
                test_losses.append(test_loss)
                print("Test fold: {} | Test loss: {:.2f} | Pearson Coef: {:.2f}".format(
                    test_fold, test_loss, pc))
        print("*" * n_delimiter + '\n')
    mu, sigma = calc_stat(test_losses)
    print("MSE: {:.2f} ± {:.2f}".format(mu, sigma))
    lo, hi = conf_inv(mu, sigma, len(test_losses))
    print("Confidence interval: [{:.2f}, {:.2f}]".format(lo, hi))
    rmse_loss = [x ** 0.5 for x in test_losses]
    mu, sigma = calc_stat(rmse_loss)
    print("RMSE: {:.2f} ± {:.2f}".format(mu, sigma))
    mu, sigma = calc_stat(pearson_coefs)
    print("Pearson: {:.2f} ± {:.2f}".format(mu, sigma))

    # with open(os.path.join(out_dir, 'y_preds.pkl'), 'wb') as f:
    #     pickle.dump(y_preds, f)
    # with open(os.path.join(out_dir, 'y_trues.pkl'), 'wb') as f:
    #     pickle.dump(y_trues, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mdl_dir', type=str, help="model dir")
    args = parser.parse_args()
    mdl_dir = os.path.join(OUTPUT_DIR, args.mdl_dir)
    calc_metrics(mdl_dir)


if __name__ == '__main__':
    main()
