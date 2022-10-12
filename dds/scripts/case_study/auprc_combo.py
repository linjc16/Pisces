import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import pubchempy as pcp
from scipy import stats
from rdkit import Chem
from matplotlib import pyplot as plt
from rdkit.Chem import rdMolDescriptors
import seaborn as sns
import pdb

diff_path = 'dds/scripts/case_study/raw_output/diff.csv'
drug_types_path = 'dds/scripts/case_study/raw_output/drug_types.csv'

MEDIUM_SIZE = 8
SMALLER_SIZE = 6
BIGGER_SIZE = 25
plt.rc('font', family='Helvetica', size=MEDIUM_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALLER_SIZE)  # fontsize of the tick labels
plt.rc('figure', titlesize=MEDIUM_SIZE)
plt.rc('legend', fontsize=MEDIUM_SIZE)
FIG_HEIGHT = 2
FIG_WIDTH = 2
plt.style.use('dark_background')

def main():
    diff = pd.read_csv(diff_path)
    drug_types = pd.read_csv(drug_types_path)
    drug_types_dict = dict(zip(drug_types['drug_names'].tolist(), drug_types['drug_types'].tolist()))

    top_combos = 75
    improvs = diff['AUPRC_diff'].to_numpy()[top_combos:]
    # pdb.set_trace()
    
    combo_types = []
    for i in range(len(diff)):
        data_curr = diff.iloc[i, :]
        if drug_types_dict[data_curr['anchor_names']] == drug_types_dict[data_curr['library_names']]:
            combo_types.append(1)
        else:
            combo_types.append(0)
    
    combo_types = np.array(combo_types)[top_combos:]

    improv_homo = improvs[combo_types == 1]
    improv_hetero = improvs[combo_types == 0]
    
    pval1 = stats.ttest_ind(improv_homo, improv_hetero, alternative='greater')
    print('Hypothethis suitable modaliry is greater p value: {}'.format(pval1))

    plt.clf()
    palette1 = ["#7fc97f", "#beaed4", "#fdc086", "#ffff99", "#386cb0"]
    group, x_jitter = [], []
    g1, g2 = [], []

    for i in range(len(combo_types)):
        if combo_types[i] == 1:
            g1.append(improvs[i])
            group.append('Suitable')
        else:
            g2.append(improvs[i])
            group.append('Not suitable')
    
    order = ['Suitable', 'Not suitable']
    sns.set_palette(palette=palette1)
    fig, ax = plt.subplots(figsize=(1*FIG_WIDTH, FIG_HEIGHT))
    ax = sns.violinplot(x=group, y=improvs, order=order)
    #plt.scatter(x_jitter, diameters, c='k', s=1.2, alpha=0.5)
    ax.set_xlabel('Suitable modaility combos')
    ax.set_ylabel('AUPRC Improvements')
    #ax.set_yticklabels(labels=['Top 50 Drug Combo \n With Improvements', 'Others'], fontsize=MEDIUM_SIZE)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tight_layout()
    plt.rcParams['savefig.dpi'] = 800
    plt.savefig('suitable.png')

if __name__ == '__main__':
    main()