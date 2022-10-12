import numpy as np
import pdb
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy import stats
import statsmodels.api as sm


import plot_settings
import plot_utils

num_fold = 5

root_dir = 'dds/scripts/visualization/raw_output'

g_bacc = []
t_bacc = []
g_auprc = []
t_auprc = []
g_f1 = []
t_f1 = []
g_kappa = []
t_kappa = []



for i in range(num_fold):
    g_results = pd.read_csv(os.path.join(root_dir, f'cell_level_drug_gcn_base_fold{i}.csv'))
    t_results = pd.read_csv(os.path.join(root_dir, f'cell_level_drug_transfomer_base_fold{i}.csv'))

    g_bacc.append([g_results['bacc'].tolist()])
    t_bacc.append([t_results['bacc'].tolist()])
    g_auprc.append([g_results['auprc'].tolist()])
    t_auprc.append([t_results['auprc'].tolist()])
    g_f1.append([g_results['f1'].tolist()])
    t_f1.append([t_results['f1'].tolist()])
    g_kappa.append([g_results['kappa'].tolist()])
    t_kappa.append([t_results['kappa'].tolist()])

# pdb.set_trace()
g_bacc = np.concatenate(g_bacc, axis=0).mean(axis=0)
t_bacc = np.concatenate(t_bacc, axis=0).mean(axis=0)

g_auprc = np.concatenate(g_auprc, axis=0).mean(axis=0)
t_auprc = np.concatenate(t_auprc, axis=0).mean(axis=0)

g_f1 = np.concatenate(g_f1, axis=0).mean(axis=0)
t_f1 = np.concatenate(t_f1, axis=0).mean(axis=0)

g_kappa = np.concatenate(g_kappa, axis=0).mean(axis=0)
t_kappa = np.concatenate(t_kappa, axis=0).mean(axis=0)


def linear_regression(x, y, name, x_aixs_name, y_axis_name, x_lim=(0, 1), y_lim=(0, 1), text_xy=(0.5, 0.5)):

    cl=0.95
    x, y = x[~np.isnan(x) * ~np.isnan(y)], y[~np.isnan(x) * ~np.isnan(y)]
    data = pd.DataFrame({'x' : x, 'y' : y}).sort_values(by='x')
    x = np.asarray(data['x'])
    y = np.asarray(data['y'])
    
    alpha = 1 - cl
    
    a, b = np.polyfit(x, y, deg=1)

    def fitted_line(x):
        return a * x + b
    
    Sxx = np.sum((x-np.mean(x))**2)
    Syy = np.sum((y-np.mean(y))**2)
    Sxy = np.sum((x-np.mean(x))*(y-np.mean(y)))
    
    dof = len(x) - 2
    
    sigma = np.sqrt((Syy - a*Sxy)/dof)
    
    R_sq = a*Sxy/Syy
    
    t_value = stats.t.isf(alpha/2, dof)
    

    X2 = sm.add_constant(x)
    est = sm.OLS(y, X2)
    est2 = est.fit()
    print(est2.summary())

    p_value = est2.pvalues[1]

    ax = plot_settings.get_square_axis(4, 4)
    ax.set_xlim(x_lim[0], x_lim[1])
    ax.set_ylim(y_lim[0], y_lim[1])
    
    plot_utils.scatter_plot(ax, xs=x, ys=y, color='white',
                        xlabel=x_aixs_name, ylabel=y_axis_name,
                        size=15, edge_color='gray') # , label='cell lines' 
    
    pve_text = f'FVE = {round(R_sq, 2)}\n$P$ = '
    p_text = f'{p_value:.2e}'
    
    
    idx_e = p_text.index('e')
    power = int(p_text[idx_e + 1:])
    p_text = p_text[:idx_e] + rf' $\times 10^'+'{'+str(power)+'}$'
    

    x = x[x > 0.05]

    sns.lineplot(x=x, y=fitted_line(x), lw=2) # label=f'Linear regression'
    sns.lineplot(x=x, y=fitted_line(x) + t_value*sigma*np.sqrt(1./len(x)+(x-np.mean(x))**2/Sxx), lw=2, color='g', linestyle='--')
    sns.lineplot(x=x, y=fitted_line(x) - t_value*sigma*np.sqrt(1./len(x)+(x-np.mean(x))**2/Sxx), lw=2, color='g', linestyle='--')
    
    
    ax.fill_between(
        x, 
        fitted_line(x) - t_value*sigma*np.sqrt(1./len(x)+(x-np.mean(x))**2/Sxx), 
        fitted_line(x) + t_value*sigma*np.sqrt(1./len(x)+(x-np.mean(x))**2/Sxx), 
        alpha=0.3,
        label=r'95% confidence band'
        )

    plt.tight_layout()
    plot_utils.format_ax(ax)
    plt.savefig(os.path.join('./', f'vis_{name}.pdf'), dpi=300)

    

linear_regression(g_bacc, t_bacc, 'BACC', 'BACC (Molecular-graph-based)', 'BACC (SMILES-based)', (0.45, 0.9), (0.45, 0.9), (0.7, 0.5))
linear_regression(g_auprc, t_auprc, 'AUPRC', 'AUPRC (Molecular-graph-based)', 'AUPRC (SMILES-based)', (0, 0.8), (0, 0.8), (0.5, 0.15))
linear_regression(g_f1, t_f1, 'F1 scores', 'F1 scores (Molecular-graph-based)', 'F1 scores (SMILES-based)', (0, 0.7), (0, 0.7), (0.4, 0.15))
linear_regression(g_kappa, t_kappa, 'KAPPA', 'KAPPA (Molecular-graph-based)', 'KAPPA (SMILES-based)', (0, 0.7), (0, 0.7), (0.4, 0.15))