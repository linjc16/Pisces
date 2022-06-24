import numpy as np
import torch
from torch.utils.data import Dataset

from typing import List


class C2VDataset(Dataset):

    def __init__(self, cell_tgt_file: str, valid_node_file: str):
        tgt = np.load(cell_tgt_file)
        nodes = np.load(valid_node_file)
        self.tgt = torch.from_numpy(tgt).float()
        self.node_indices = torch.from_numpy(nodes)

    def __len__(self):
        return self.tgt.shape[0]

    def __getitem__(self, item):
        return torch.tensor(item, dtype=torch.long), self.tgt[item]


class C2VSymDataset(Dataset):

    def __init__(self, target_files: List[str], node_files: List[str]):
        self.targets = []
        self.nodes = []
        for t_f, n_f in zip(target_files, node_files):
            t = np.load(t_f)
            t = torch.from_numpy(t).float()
            n = np.load(n_f)
            n = torch.from_numpy(n)
            self.targets.append(t)
            self.nodes.append(n)

    def __len__(self):
        return self.targets[0].shape[0]

    def __getitem__(self, item):
        ret = [target[item] for target in self.targets]
        ret.insert(0, torch.tensor(item, dtype=torch.long))
        return tuple(ret)
