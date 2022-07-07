import argparse
from operator import index
import os
import os.path as osp
import random
import logging

from datetime import datetime
from time import perf_counter as t

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import pdb
from tqdm import tqdm
import pandas as pd

from torch.utils.data import DataLoader

from const import DATA_DIR
from model import AutoEncoder
from utils import save_args, save_best_model, save_and_visual_loss, find_best_model

def train_step(model, batch):
    model.train()
    optimizer.zero_grad()
    y_pred, _ = model(batch)
    step_loss = loss_func(y_pred, batch)
    step_loss.backward()
    optimizer.step()

    return step_loss.item()

def eval_step(model, batch):
    model.eval()
    with torch.no_grad():
        y_pred, _ = model(batch)
        step_loss = loss_func(y_pred, batch)
    # pdb.set_trace()

    return step_loss.item()

def gen_cell_emb(model, data):
    model.eval()
    with torch.no_grad():
        _, emb = model(data)
    return emb.cpu().numpy()

if __name__ == '__main__':
    time_str = str(datetime.now().strftime('%y%m%d%H%M'))
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss', type=str, choices=['mse', 'bce'], default='mse')
    parser.add_argument('--lr', type=float, default=1e-3, help="initial learning rate")
    parser.add_argument('--epoch', type=int, default=1000, help="number of epochs to train.")
    parser.add_argument('--batch', type=int, default=2, help="batch size")
    parser.add_argument('--keep', type=int, default=1, help="max number of best models to keep")
    parser.add_argument('--patience', type=int, default=50, help="patience")
    parser.add_argument('--gpu', type=int, default=None,
                        help="gpu id to use")
    parser.add_argument('--data_dir', type=str, default='data/cell_tpm.csv')
    parser.add_argument('--suffix', type=str, default=time_str, help="suffix for model dir")
    parser.add_argument('--type', type=str, choices=['exp', 'mut'], default='exp')
    parser.add_argument('--hidden_size', type=int, default=8192)
    args = parser.parse_args()

    mdl_dir = osp.join(DATA_DIR, 'mdl_{}_{}'.format(args.type, args.suffix))
    loss_file = osp.join(mdl_dir, 'loss.pkl')
    if not osp.exists(mdl_dir):
        os.makedirs(mdl_dir)
    save_args(args, osp.join(mdl_dir, 'args.json'))

    log_file = os.path.join(mdl_dir, 'train.log')
    logging.basicConfig(filename=log_file,
                        format='%(asctime)s %(message)s',
                        datefmt='[%Y-%m-%d %H:%M:%S]',
                        level=logging.INFO)

    torch.manual_seed(23333)
    random.seed(12345)
    learning_rate = args.lr
    batch_size = args.batch

    num_epochs = args.epoch
    weight_decay = 1e-5

    if torch.cuda.is_available() and args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        device = torch.device('cuda:{}'.format(args.gpu))
    else:
        device = torch.device('cpu')
    
    data_df = pd.read_csv(args.data_dir, index_col=0)
    cell_names = data_df['cell_line_names'].tolist()
    CELL_TO_INDEX_DICT = {cell: id for id, cell in enumerate(cell_names)}

    cell2id_df = pd.DataFrame.from_dict(
        {
            'cell': CELL_TO_INDEX_DICT.keys(),
            'id': CELL_TO_INDEX_DICT.values()
        }
    )
    cell2id_df.to_csv('baselines/AuDNNsynergy/cell/data_ours/cell2id.tsv', sep='\t', index=False, header=True)

    data = data_df.iloc[:, 1:].to_numpy()
    pdb.set_trace()
    data = np.log2(data + 1e-3)
    idx_full = np.arange(len(data))
    np.random.shuffle(idx_full)

    len_valid = int(len(data) * 0.1)
    valid_idx = idx_full[0:len_valid]
    train_idx = np.delete(idx_full, np.arange(0, len_valid))

    data_train = torch.FloatTensor(data[train_idx.tolist()])
    data_valid = torch.FloatTensor(data[valid_idx.tolist()])
    

    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(data_valid, batch_size=batch_size, shuffle=False)

    model = AutoEncoder(data_train.size(1), args.hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if args.loss == 'mse':
        loss_func = nn.MSELoss(reduction='mean')
    else:
        loss_func = nn.BCEWithLogitsLoss(reduction='mean')

    logging.info("Check model.")
    logging.info(model)

    logging.info("Start training.")
    losses = []
    min_loss = float('inf')
    angry = 0

    start = t()
    prev = start

    for epoch in tqdm(range(1, num_epochs + 1)):
        epoch_loss = 0
        now = t()
        for step, batch in enumerate(train_loader):
            batch = batch.to(device)
            loss = train_step(model, batch)
            epoch_loss += loss * len(batch)
        epoch_loss /= len(data_train)
        logging.info('Epoch={:04d} Training Loss={:.4f}'.format(epoch, epoch_loss))
        
        prev = now

        # eval
        epoch_loss_eval = 0
        for step, batch in enumerate(valid_loader):
            batch = batch.to(device)
            loss = eval_step(model, batch)
            epoch_loss_eval += loss * len(batch)
        
        epoch_loss_eval /= len(data_valid)
        losses.append(epoch_loss_eval)
        logging.info('Epoch={:04d} Valid Loss={:.4f}'.format(epoch, epoch_loss_eval))

        if epoch_loss_eval < min_loss:
            min_loss = epoch_loss_eval
            save_best_model(model.state_dict(), mdl_dir, epoch, args.keep)
            angry = 0
        else:
            angry += 1
        if angry == args.patience:
            break
    logging.info("Training completed.")
    logging.info("Min train loss: {:.4f} | Epoch: {:04d}".format(min_loss, losses.index(min_loss)))
    logging.info("Save to {}".format(mdl_dir))

    save_and_visual_loss(losses, loss_file, title='Train Loss', xlabel='epoch', ylabel='Loss')
    logging.info("Save train loss curve to {}".format(loss_file))

    model.load_state_dict(torch.load(find_best_model(mdl_dir), map_location=torch.device('cpu')))
    embeddings = gen_cell_emb(model, torch.FloatTensor(data).to(device))
    np.save(os.path.join(mdl_dir, 'embeddings.npy'), embeddings)
    logging.info("Save {}".format(os.path.join(mdl_dir, 'embeddings.npy')))
    # pdb.set_trace()