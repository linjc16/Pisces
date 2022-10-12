import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pdb

import plot_settings_bar
import plot_utils

trans_DeepDDS = [
    [[0.6683, 0.4433, 0.4271, 0.4022]],
    [[0.6579, 0.3953, 0.3978, 0.3713]],
    [[0.6377, 0.4034, 0.3737, 0.3495]],
    [[0.6384, 0.3980, 0.3723, 0.3475]],
    [[0.6662, 0.3933, 0.4048, 0.3774]]]
trans_GraphSyn = [
    [[0.6477, 0.4800, 0.4028, 0.3815]],
    [[0.6397, 0.4510, 0.3801, 0.3581]],
    [[0.6417, 0.4611, 0.3833, 0.3614]],
    [[0.6369, 0.4566, 0.3774, 0.3560]],
    [[0.6381, 0.4266, 0.3681, 0.3445]]]
trans_PRODeepSyn = [
    [[0.6256, 0.3637, 0.3378, 0.3108]],
    [[0.6243, 0.3133, 0.3200, 0.2898]],
    [[0.6164, 0.3644, 0.3282, 0.3036]],
    [[0.6236, 0.3236, 0.3191, 0.2885]],
    [[0.6214, 0.3090, 0.3121, 0.2815]]]
trans_DeepSynergy = [
    [[0.5022, 0.2000, 0.0092, 0.0081]],
    [[0.5121, 0.2289, 0.0485, 0.0442]],
    [[0.5352, 0.2458, 0.1290, 0.1151]],
    [[0.5, 0.1872, 0, 0]],
    [[0.5285, 0.2351, 0.1077, 0.0973]]
]
trans_AuDNNSyn = [
    [[0.6071, 0.2426, 0.2787, 0.2451]],
    [[0.6674, 0.2833, 0.3465, 0.3075]],
    [[0.6207, 0.2536, 0.2941, 0.2580]],
    [[0.6284, 0.2515, 0.3024, 0.2653]],
    [[0.6221, 0.2436, 0.2899, 0.2530]]
]
trans_ours = [
    [[0.7168, 0.4874, 0.4865, 0.4599]],
    [[0.6961, 0.4718, 0.4692, 0.4445]],
    [[0.6994, 0.4749, 0.4721, 0.4467]],
    [[0.7034, 0.4919, 0.4816, 0.4569]],
    [[0.6879, 0.4316, 0.4538, 0.4288]]]

models_trans = ['DeepSynergy', 'AuDNNsynergy', 'PRODeepSyn', 'GraphSynergy', 'DeepDDS', 'Ours']
trans = (trans_DeepSynergy, trans_AuDNNSyn, trans_PRODeepSyn, trans_GraphSyn, trans_DeepDDS, trans_ours)

lc_DeepDDS = [
    [[0.5882, 0.2093, 0.2447, 0.2154]],
    [[0.6283, 0.2374, 0.2992, 0.2690]],
    [[0.6073, 0.2497, 0.2893, 0.2606]]
]

lc_PRODeepSyn = [
    [[0.5571, 0.2060, 0.1894, 0.1705]],
    [[0.5793, 0.1991, 0.2232, 0.1973]],
    [[0.5688, 0.2236, 0.2152, 0.1917]]
]


lc_DeepSynergy = [
    [[0.5, 0.0809, 0, 0]],
    [[0.5, 0.0794, 0, 0]],
    [[0.5, 0.1233, 0, 0]]
]

lc_AuDnnSyn = [
    [[0.5533, 0.2339, 0.1866, 0.1737]],
    [[0.5543, 0.1636, 0.1735, 0.1469]],
    [[0.5521, 0.2737, 0.1793, 0.1627]]
]

lc_Ours = [
    [[0.6210, 0.2347, 0.2961, 0.2635]],
    [[0.6591, 0.2376, 0.3186, 0.2839]],
    [[0.6104, 0.2340, 0.2789, 0.2454]]
]

lc_GraphSyn = [
    [[0, 0, 0, 0]],
    [[0, 0, 0, 0]],
    [[0, 0, 0, 0]],
]

models_lc = ['DeepSynergy', 'AuDNNsynergy', 'PRODeepSyn', 'GraphSynergy', 'DeepDDS', 'Ours']
lc = (lc_DeepSynergy, lc_AuDnnSyn, lc_PRODeepSyn, lc_GraphSyn, lc_DeepDDS, lc_Ours)

ldc_DeepDDS = [
    [[0.6305, 0.3144, 0.3459, 0.3204]],
    [[0.6003, 0.3524, 0.3019, 0.2795]],
    [[0.5960, 0.2804, 0.2751, 0.2534]]
]

ldc_PRODeepSyn = [
    [[0.6103, 0.2675, 0.2920, 0.2629]],
    [[0.6183, 0.3006, 0.3091, 0.2772]],
    [[0.6035, 0.2239, 0.2636, 0.2345]]
]

ldc_AuDNNSyn = [
    [[0.5753, 0.2545, 0.2283, 0.2074]],
    [[0.5474, 0.2691, 0.1653, 0.1524]],
    [[0.5466, 0.1500, 0.1478, 0.1214]]
]

ldc_GraphSyn = [
    [[0.6116, 0.2977, 0.2913, 0.2626]],
    [[0.6409, 0.3167, 0.3299, 0.2950]],
    [[0.6019, 0.2901, 0.2728, 0.2492]]
]

ldc_DeepSynergy = [
    [[0.5, 0.1602, 0, 0]],
    [[0.5, 0.1942, 0, 0]],
    [[0.5, 0.1301, 0, 0]]
]

ldc_Ours = [
    [[0.6583, 0.3522, 0.3839, 0.3562]],
    [[0.6707, 0.3977, 0.4036, 0.3730]],
    [[0.6767, 0.3632, 0.4007, 0.3748]]
]

models_ldc = ['DeepSynergy', 'AuDNNsynergy', 'PRODeepSyn', 'GraphSynergy', 'DeepDDS', 'Ours']
ldc = (ldc_DeepSynergy, ldc_AuDNNSyn, ldc_PRODeepSyn, ldc_GraphSyn, ldc_DeepDDS, ldc_Ours)


settings = ['5-fold CV', 'Stratified CV for\ndrug combinations', 'Stratified CV for\ncell lines']



def bar_plot(dataset, models, labels, name, metrics_id):

    
    ax = plot_settings_bar.get_wider_axis(3, 6)

    colors = [plot_settings_bar.get_model_colors(mod) for mod in models]
    
    means = []
    stderrs = []

    for data in dataset:
        task_vals = []
        for result in data:
            result = np.concatenate(result)
            task_vals.append((np.mean(result[:, metrics_id]), np.std(result[:, metrics_id]) / np.sqrt(result.shape[0])))
        means.append([v[0] for v in task_vals])
        stderrs.append([v[1] for v in task_vals])

    min_val = [0.45, 0.0, 0, 0]

    plot_utils.grouped_barplot_graphsyn(
        ax, means, 
        settings,
        xlabel='', ylabel=name, color_legend=labels,
        nested_color=colors, nested_errs=stderrs, tickloc_top=False, rotangle=35, anchorpoint='right',
        min_val=min_val[metrics_id])
    
    plot_utils.format_ax(ax)

    plot_utils.format_legend(ax, *ax.get_legend_handles_labels(), loc='upper right', 
                            ncols=1)
    
    plot_utils.put_legend_outside_plot(ax, anchorage=(1.8, 1.01))

    
    # plt.savefig(f'bar_{name}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'bar_{name}.pdf', dpi=300, bbox_inches='tight')

dataset = (trans, ldc, lc)

bar_plot(dataset, models_trans, models_trans, 'BACC', 0)
bar_plot(dataset, models_trans, models_trans, 'AUPRC', 1)
bar_plot(dataset, models_trans, models_trans, 'F1', 2)
bar_plot(dataset, models_trans, models_trans, 'KAPPA', 3)