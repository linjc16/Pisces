from sklearn.manifold import TSNE
import numpy as np
import pdb
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

import plot_settings
import plot_utils

root_dir = 'dds/scripts/visualization/raw_output'

tri_pair_feats = torch.load(os.path.join(root_dir, 'tri_pairs_feat_GS.pt')).numpy()
labels = torch.load(os.path.join(root_dir, 'labels_GS.pt')).long().squeeze(1).numpy()


labels_pos, labels_neg = labels[labels == 1], labels[labels == 0]
tri_pair_feats_pos, tri_pair_feats_neg = tri_pair_feats[labels == 1], tri_pair_feats[labels == 0]


dw_neg_num = 2000

labels_neg = labels_neg[0:dw_neg_num]
tri_pair_feats_neg = tri_pair_feats_neg[0:dw_neg_num]

neg_num = labels_neg.shape[0]
pos_num = labels_pos.shape[0]

labels_curr = np.concatenate([labels_pos, labels_neg], axis=0)
tri_pair_feats_curr = np.concatenate([tri_pair_feats_pos, tri_pair_feats_neg], axis=0)

features = tri_pair_feats_curr
tsne = TSNE(n_components=2, init='pca', random_state=14545)
class_num = len(np.unique(labels_curr))
tsne_features = tsne.fit_transform(features) 

ax = plot_settings.get_double_square_axis()

colors = ['#33a02c', '#a6cee3']

point_size = 2

plot_utils.scatter_plot(
    ax, xs=tsne_features[-neg_num:, 0], ys=tsne_features[-neg_num:, 1], color=colors[0],
    xlabel='', ylabel='', size=point_size
) 

lp0 = plot_utils.scatter_plot(
    ax, xs=[], ys=[], color=colors[0],
    xlabel='', ylabel='', size=50, label='0'
) 

plot_utils.scatter_plot(
    ax, xs=tsne_features[:pos_num, 0], ys=tsne_features[:pos_num, 1], color=colors[1],
    xlabel='', ylabel='', size=point_size
)

lp1 = plot_utils.scatter_plot(
    ax, xs=[], ys=[], color=colors[1],
    xlabel='', ylabel='', size=50, label='1'
)

handles=[lp0, lp1]
plot_utils.format_legend(ax, handles, ['0', '1'])
plot_utils.put_legend_outside_plot(ax)
plot_utils.format_ax(ax)
plt.axis('off')
plt.savefig(os.path.join('output', 'vis_graphsy.png'))