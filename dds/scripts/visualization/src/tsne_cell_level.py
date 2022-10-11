from cProfile import label
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

import plot_settings
import plot_utils

root_dir = 'dds/scripts/visualization/raw_output'

tri_pair_feats = np.load(os.path.join(root_dir, 'tri.npy'))
cells = np.load(os.path.join(root_dir, 'cells.npy')).squeeze(1)
labels = np.load(os.path.join(root_dir, 'labels.npy')).squeeze(1)

cell_names = pd.read_csv('data/cell_tpm.csv')['cell_line_names']


cell_lines = [88]

for cell_line in cell_lines:
    tri_pair_feats_curr_ori = tri_pair_feats[cells == cell_line]
    labels_curr_ori = labels[cells == cell_line]
    
    labels_pos, labels_neg = labels_curr_ori[labels_curr_ori == 1], labels_curr_ori[labels_curr_ori == 0]
    tri_pair_feats_pos, tri_pair_feats_neg = tri_pair_feats_curr_ori[labels_curr_ori == 1], tri_pair_feats_curr_ori[labels_curr_ori == 0]

    dw_neg_num = 100

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

    colors = ['#d8b365', '#5ab4ac']

    point_size = 30
    
    plot_utils.scatter_plot(
        ax, xs=tsne_features[-neg_num:, 0], ys=tsne_features[-neg_num:, 1], color=colors[0],
        xlabel='', ylabel='', size=point_size
    ) 

    lp0 = plot_utils.scatter_plot(
        ax, xs=[], ys=[], color=colors[0],
        xlabel='', ylabel='', size=100, label='No synergy'
    ) 

    plot_utils.scatter_plot(
        ax, xs=tsne_features[:pos_num, 0], ys=tsne_features[:pos_num, 1], color=colors[1],
        xlabel='', ylabel='', size=point_size
    )

    lp1 = plot_utils.scatter_plot(
        ax, xs=[], ys=[], color=colors[1],
        xlabel='', ylabel='', size=100, label='Synergy'
    )

    handles=[lp0, lp1]
    plot_utils.format_legend(ax, handles, ['0', '1'])
    plot_utils.put_legend_outside_plot(ax)

    plot_utils.format_ax(ax)
    plt.axis('off')
    plt.savefig(os.path.join('output', f'vis_{cell_line}.pdf'), dpi=300)
