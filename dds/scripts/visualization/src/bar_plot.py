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
trans_ours = [
    [[0.7168, 0.4874, 0.4865, 0.4599]],
    [[0.6961, 0.4718, 0.4692, 0.4445]],
    [[0.6994, 0.4749, 0.4721, 0.4467]],
    [[0.7034, 0.4919, 0.4816, 0.4569]],
    [[0.6879, 0.4316, 0.4538, 0.4288]]]

models_trans = ['PRODeepSyn', 'GraphSynergy', 'DeepDDS', 'Ours']
trans = (trans_PRODeepSyn, trans_GraphSyn, trans_DeepDDS, trans_ours)

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

models_lc = ['PRODeepSyn', 'GraphSynergy', 'DeepDDS', 'Ours']
lc = (lc_PRODeepSyn, lc_GraphSyn, lc_DeepDDS, lc_Ours)

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

ldc_GraphSyn = [
    [[0.6116, 0.2977, 0.2913, 0.2626]],
    [[0.6409, 0.3167, 0.3299, 0.2950]],
    [[0.6019, 0.2901, 0.2728, 0.2492]]
]

ldc_Ours = [
    [[0.6583, 0.3522, 0.3839, 0.3562]],
    [[0.6707, 0.3977, 0.4036, 0.3730]],
    [[0.6767, 0.3632, 0.4007, 0.3748]]
]

models_ldc = ['PRODeepSyn', 'GraphSynergy', 'DeepDDS', 'Ours']
ldc = (ldc_PRODeepSyn, ldc_GraphSyn, ldc_DeepDDS, ldc_Ours)


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

    min_val = [0.55, 0.2, 0.2, 0.15]

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