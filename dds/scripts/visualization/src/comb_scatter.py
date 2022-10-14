import numpy as np
import pdb
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy import stats
import statsmodels.api as sm
from collections import defaultdict


import plot_settings
import plot_utils

root_dir = 'dds/scripts/visualization/raw_output_scatter'

g_results = pd.read_csv(os.path.join(root_dir, f'comb_level_trans_results_graph.csv'))
t_results = pd.read_csv(os.path.join(root_dir, f'comb_level_trans_results_smiles.csv'))

drug_names = pd.read_csv('data/drug_smiles.csv')['drug_names'].tolist()


g_bacc_dict = defaultdict(list)
t_bacc_dict = defaultdict(list)
g_auprc_dict = defaultdict(list)
t_auprc_dict = defaultdict(list)
g_f1_dict = defaultdict(list)
t_f1_dict = defaultdict(list)
g_kappa_dict = defaultdict(list)
t_kappa_dict = defaultdict(list)


for i in range(len(g_results)):

    data_g = g_results.iloc[i, :]
    data_t = t_results.iloc[i, :]

    g_bacc_dict[data_g['anchor_names']].append(data_g['BACC'])
    g_bacc_dict[data_g['library_names']].append(data_g['BACC'])
    t_bacc_dict[data_t['anchor_names']].append(data_t['BACC'])
    t_bacc_dict[data_t['library_names']].append(data_t['BACC'])

    if not np.isnan(data_g['AUPRC']): 
        g_auprc_dict[data_g['anchor_names']].append(data_g['AUPRC'])
        g_auprc_dict[data_g['library_names']].append(data_g['AUPRC'])
    
    if not np.isnan(data_t['AUPRC']): 
        t_auprc_dict[data_t['anchor_names']].append(data_t['AUPRC'])
        t_auprc_dict[data_t['library_names']].append(data_t['AUPRC'])
    
    g_f1_dict[data_g['anchor_names']].append(data_g['F1'])
    g_f1_dict[data_g['library_names']].append(data_g['F1'])
    t_f1_dict[data_t['anchor_names']].append(data_t['F1'])
    t_f1_dict[data_t['library_names']].append(data_t['F1'])

    if not np.isnan(data_g['KAPPA']):
        g_kappa_dict[data_g['anchor_names']].append(data_g['KAPPA'])
        g_kappa_dict[data_g['library_names']].append(data_g['KAPPA'])
    if not np.isnan(data_t['KAPPA']):
        t_kappa_dict[data_t['anchor_names']].append(data_t['KAPPA'])
        t_kappa_dict[data_t['library_names']].append(data_t['KAPPA'])

g_bacc = []
t_bacc = []
g_auprc = []
t_auprc = []
g_f1 = []
t_f1 = []
g_kappa = []
t_kappa = []




for drug_name in drug_names:
    # pdb.set_trace()
    g_bacc.append(np.array(g_bacc_dict[drug_name]).mean())
    t_bacc.append(np.array(t_bacc_dict[drug_name]).mean())

    g_auprc.append(np.array(g_auprc_dict[drug_name]).mean())
    t_auprc.append(np.array(t_auprc_dict[drug_name]).mean())

    g_f1.append(np.array(g_f1_dict[drug_name]).mean())
    t_f1.append(np.array(t_f1_dict[drug_name]).mean())

    g_kappa.append(np.array(g_kappa_dict[drug_name]).mean())
    t_kappa.append(np.array(t_kappa_dict[drug_name]).mean())


g_bacc = np.array(g_bacc)
t_bacc = np.array(t_bacc)

g_auprc = np.array(g_auprc)
t_auprc = np.array(t_auprc)

g_f1 = np.array(g_f1)
t_f1 = np.array(t_f1)

g_kappa = np.array(g_kappa)
t_kappa = np.array(t_kappa)

# pdb.set_trace()

# split the drugs
drug_types = (g_kappa > t_kappa).astype(np.int64)


drug_types_df = pd.DataFrame.from_dict(
    {
        'drug_names': drug_names,
        'drug_types': drug_types.tolist()
    }
)

drug_types_df.to_csv('drug_types.csv', index=False)

pdb.set_trace()

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
    
    print(f'p_value: {p_value}')

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
    plt.savefig(os.path.join('./', f'vis_comb_{name}.pdf'), dpi=300)


linear_regression(g_bacc, t_bacc, 'BACC', 'BACC (Molecular-graph-based)', 'BACC (SMILES-based)', (0.55, 1), (0.55, 1), (0.7, 0.5))
linear_regression(g_auprc, t_auprc, 'AUPRC', 'AUPRC (Molecular-graph-based)', 'AUPRC (SMILES-based)', (0, 0.6), (0, 0.6), (0.5, 0.15))
linear_regression(g_f1, t_f1, 'F1 scores', 'F1 scores (Molecular-graph-based)', 'F1 scores (SMILES-based)', (0, 0.45), (0, 0.45), (0.4, 0.15))
linear_regression(g_kappa, t_kappa, 'KAPPA', 'KAPPA (Molecular-graph-based)', 'KAPPA (SMILES-based)', (0, 0.35), (0, 0.35), (0.4, 0.15))