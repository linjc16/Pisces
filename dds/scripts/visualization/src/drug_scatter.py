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

g_results = pd.read_csv(os.path.join(root_dir, f'drug_level_trans_results_graph.csv'))
t_results = pd.read_csv(os.path.join(root_dir, f'drug_level_trans_results_smiles.csv'))

drug_names = pd.read_csv('data/drug_smiles.csv')['drug_names'].tolist()


threshold = 0.05

count = 0

def drug_gt_ttest(g_metric, t_metic, metric_name):
    pval1 = stats.ttest_ind(g_metric, t_metic, alternative='two-sided')
    if pval1.pvalue < threshold:
        print(f'Hypothethis metrics {metric_name} SMILES and GRAPH is distinctive p value: {pval1}')
    return pval1.pvalue


g_bacc = g_results['BACC']
t_bacc = t_results['BACC']

g_auprc = g_results['AUPRC']
t_auprc = t_results['AUPRC']

g_f1 = g_results['F1']
t_f1 = t_results['F1']

g_kappa = g_results['KAPPA']
t_kappa = t_results['KAPPA']

drug_types = (g_kappa > t_kappa).astype(np.int64)


drug_types_df = pd.DataFrame.from_dict(
    {
        'drug_names': drug_names,
        'drug_types': drug_types.tolist()
    }
)

drug_types_df.to_csv('drug_types.csv', index=False)

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
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plot_utils.format_ax(ax)
    plt.savefig(os.path.join('./', f'vis_drug_{name}.pdf'), dpi=300)


linear_regression(g_bacc, t_bacc, 'BACC', 'BACC (Molecular-graph-based)', 'BACC (SMILES-based)', (0.5, 0.8), (0.5, 0.8), (0.7, 0.5))
linear_regression(g_auprc, t_auprc, 'AUPRC', 'AUPRC (Molecular-graph-based)', 'AUPRC (SMILES-based)', (0, 0.7), (0, 0.7), (0.5, 0.15))
linear_regression(g_f1, t_f1, 'F1 scores', 'F1 score (Molecular-graph-based)', 'F1 score (SMILES-based)', (0, 0.7), (0, 0.7), (0.4, 0.15))
linear_regression(g_kappa, t_kappa, 'KAPPA', 'KAPPA (Molecular-graph-based)', 'KAPPA (SMILES-based)', (0, 0.7), (0, 0.7), (0.4, 0.15))

# for drug_name in drug_names:
    # pdb.set_trace()
    # bacc
    # g_bacc_curr = np.array(g_bacc_dict[drug_name])
    # t_bacc_curr = np.array(t_bacc_dict[drug_name])
    
    # g_auprc_curr = np.array(g_auprc_dict[drug_name])
    # t_auprc_curr = np.array(t_auprc_dict[drug_name])

    # g_f1_curr = np.array(g_f1_dict[drug_name])
    # t_f1_curr = np.array(t_f1_dict[drug_name])

    # g_kappa_curr = np.array(g_kappa_dict[drug_name])
    # t_kappa_curr = np.array(t_kappa_dict[drug_name])

    # # pval_bacc = drug_gt_ttest(g_bacc_curr[g_bacc_curr > t_bacc_curr], t_bacc_curr[g_bacc_curr <= t_bacc_curr], 'bacc')
    # # pval_auprc = drug_gt_ttest(g_auprc_curr[g_auprc_curr > t_auprc_curr], t_auprc_curr[g_auprc_curr <= t_auprc_curr], 'auprc')
    # # pval_f1 = drug_gt_ttest(g_f1_curr[g_f1_curr > t_f1_curr], t_f1_curr[g_f1_curr <= t_f1_curr], 'f1')
    # # pval_kappa = drug_gt_ttest(g_kappa_curr[g_kappa_curr > t_kappa_curr], g_kappa_curr[g_kappa_curr <= t_kappa_curr], 'kappa')
    
    # pval_bacc = drug_gt_ttest(np.array(g_bacc_dict[drug_name]), np.array(t_bacc_dict[drug_name]), 'bacc')
    # pval_auprc = drug_gt_ttest(np.array(g_auprc_dict[drug_name]), np.array(t_auprc_dict[drug_name]), 'auprc')
    # pval_f1 = drug_gt_ttest(np.array(g_f1_dict[drug_name]), np.array(t_f1_dict[drug_name]), 'f1')
    # pval_kappa = drug_gt_ttest(np.array(g_kappa_dict[drug_name]), np.array(t_kappa_dict[drug_name]), 'kappa')


    # if pval_bacc < threshold or pval_auprc < threshold or pval_f1 < threshold or pval_kappa < threshold:
    #     print(f'Drug name: {drug_name}')
    #     print('*' * 30)
    #     count += 1

# print(count)
    
    
    # g_bacc.append(np.array(g_bacc_dict[drug_name]).mean())
    # t_bacc.append(np.array(t_bacc_dict[drug_name]).mean())

    # g_auprc.append(np.array(g_auprc_dict[drug_name]).mean())
    # t_auprc.append(np.array(t_auprc_dict[drug_name]).mean())

    # g_f1.append(np.array(g_f1_dict[drug_name]).mean())
    # t_f1.append(np.array(t_f1_dict[drug_name]).mean())

    # g_kappa.append(np.array(g_kappa_dict[drug_name]).mean())
    # t_kappa.append(np.array(t_kappa_dict[drug_name]).mean())



