import matplotlib.pyplot as plt
import os
import pickle
import torch
import json

from collections import defaultdict


def save_best_model(state_dict, model_dir: str, best_epoch: int, keep: int):
    save_to = os.path.join(model_dir, '{}.pkl'.format(best_epoch))
    torch.save(state_dict, save_to)
    model_files = [f for f in os.listdir(model_dir) if os.path.splitext(f)[-1] == '.pkl']
    epochs = [int(os.path.splitext(f)[0]) for f in model_files if str.isdigit(f[0])]
    outdated = sorted(epochs, reverse=True)[keep:]
    for n in outdated:
        os.remove(os.path.join(model_dir, '{}.pkl'.format(n)))


def find_best_model(model_dir: str):
    model_files = [f for f in os.listdir(model_dir) if os.path.splitext(f)[-1] == '.pkl']
    epochs = [int(os.path.splitext(f)[0]) for f in model_files if str.isdigit(f[0])]
    best_epoch = max(epochs)
    return os.path.join(model_dir, '{}.pkl'.format(best_epoch))


def save_and_visual_loss(loss: list, loss_file: str,
                         title: str = None, xlabel: str = None, ylabel: str = None, step: int = 1):
    with open(loss_file, 'wb') as f:
        pickle.dump(loss, f)

    plt.figure()
    plt.plot(range(step, step * len(loss) + 1, step), loss)
    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.savefig(loss_file[:-3] + 'png')


def save_args(args, save_to: str):
    args_dict = args.__dict__
    with open(save_to, 'w') as f:
        json.dump(args_dict, f, indent=2)


def load_args(args, load_from: str):
    args_dict = args.__dict__
    with open(load_from, 'r') as f:
        tmp_dict = json.load(f)
    for k, v in tmp_dict.items():
        args_dict[k] = v


def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.shape[1],),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0
    return x


def read_map(map_file, multi=False):
    if multi:
        d = defaultdict(list)
    else:
        d = {}
    with open(map_file, 'r') as f:
        f.readline()
        if multi:
            for line in f:
                k, v = line.rstrip().split('\t')
                d[k].append(int(v))
        else:
            for line in f:
                k, v = line.rstrip().split('\t')
                d[k] = int(v)
    return d
