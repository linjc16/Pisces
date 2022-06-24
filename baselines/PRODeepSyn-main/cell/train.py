import argparse
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
import dgl

from torch.utils.data import DataLoader

from const import COO_FILE, NODE_FEAT_FILE, DATA_DIR
from model import GCNEncoder, Cell2Vec
from dataset import C2VDataset
from utils import save_args, save_best_model, save_and_visual_loss, find_best_model


def train_step(mdl: Cell2Vec, g: dgl.DGLGraph, node_x, node_idx, cell_idx, y_true):
    mdl.train()
    optimizer.zero_grad()
    y_pred = mdl(g, node_x, node_idx, cell_idx)
    step_loss = loss_func(y_pred, y_true)
    step_loss.backward()
    optimizer.step()

    return step_loss.item()


def gen_emb(mdl: Cell2Vec):
    mdl.eval()
    with torch.no_grad():
        emb = mdl.embeddings.weight.data
    return emb.cpu().numpy()


def p_type(x):
    if isinstance(x, list):
        for xx in x:
            assert 0 <= xx < 1
    else:
        assert 0 <= x < 1
    return x


def get_graph_data():
    edges = np.load(COO_FILE).astype(int).transpose()
    eid = torch.from_numpy(edges)
    feat = np.load(NODE_FEAT_FILE)
    feat = torch.from_numpy(feat).float()
    return eid, feat


if __name__ == '__main__':
    time_str = str(datetime.now().strftime('%y%m%d%H%M'))

    parser = argparse.ArgumentParser()
    parser.add_argument('target', type=str, help="target feature file of cells")
    parser.add_argument('valid_nodes', type=str, help="list of valid nodes")
    parser.add_argument('--conv', type=int, default=2,
                        help="the number of graph conv layer, must be no less than 2")
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help="dim of hidden space for GCN layer")
    parser.add_argument('--loss', type=str, choices=['mse', 'bce'], default='mse')
    parser.add_argument('--emb_dim', type=int, default=256, help="dim of cell embeddings")
    parser.add_argument('--active', type=str, choices=['relu', 'prelu', 'elu', 'leaky'], default='relu',
                        help="activate function")
    parser.add_argument('--lr', type=float, default=1e-3, help="initial learning rate")
    parser.add_argument('--epoch', type=int, default=1000, help="number of epochs to train.")
    parser.add_argument('--batch', type=int, default=2, help="batch size")
    parser.add_argument('--keep', type=int, default=1, help="max number of best models to keep")
    parser.add_argument('--patience', type=int, default=50, help="patience")
    parser.add_argument('--gpu', type=int, default=None,
                        help="gpu id to use")
    parser.add_argument('--suffix', type=str, default=time_str, help="suffix for model dir")
    args = parser.parse_args()

    t_type = args.target.split('.')[0].split('_')[1]
    mdl_dir = osp.join(DATA_DIR, 'mdl_{}_{}x{}_{}'.format(t_type, args.hidden_dim, args.emb_dim, args.suffix))
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
    hidden_dim = args.hidden_dim
    emb_dim = args.emb_dim
    activation = ({'relu': F.relu, 'prelu': nn.PReLU(), 'elu': nn.ELU(), 'leaky': nn.LeakyReLU()})[args.active]
    num_layers = args.conv
    batch_size = args.batch

    num_epochs = args.epoch
    weight_decay = 1e-5

    edge_indices, node_features = get_graph_data()
    graph = dgl.graph((edge_indices[0], edge_indices[1]), idtype=torch.int32)
    graph = dgl.add_self_loop(graph)
    if torch.cuda.is_available() and args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        device = torch.device('cuda:{}'.format(args.gpu))
    else:
        device = torch.device('cpu')
    graph = graph.to(device)
    node_features = node_features.to(device)
    c2v_dataset = C2VDataset(osp.join(DATA_DIR, args.target), osp.join(DATA_DIR, args.valid_nodes))
    dataloader = DataLoader(c2v_dataset, shuffle=True, num_workers=2)
    node_indices = c2v_dataset.node_indices.to(device)

    encoder = GCNEncoder(node_features.shape[1], hidden_dim, activation, k=num_layers).to(device)
    model = Cell2Vec(encoder, len(c2v_dataset), emb_dim).to(device)
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
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0
        now = t()
        for step, batch in enumerate(dataloader):
            batch_x, batch_y = batch
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            loss = train_step(model, graph, node_features, node_indices, batch_x, batch_y)
            epoch_loss += loss * len(batch_x)
        epoch_loss /= len(c2v_dataset)
        logging.info('Epoch={:04d} Loss={:.4f}'.format(epoch, epoch_loss))
        losses.append(epoch_loss)
        prev = now
        if epoch_loss < min_loss:
            min_loss = epoch_loss
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
    embeddings = gen_emb(model)
    np.save(os.path.join(mdl_dir, 'embeddings.npy'), embeddings)
    logging.info("Save {}".format(os.path.join(mdl_dir, 'embeddings.npy')))
