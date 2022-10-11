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


diff_path = 'dds/scripts/case_study/raw_output/diff.csv' # in decent order

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


def get_diameter(name):
    results = pcp.get_compounds(name, 'name')[0].isomeric_smiles
    mol = Chem.MolFromSmiles(results)
    dm = Chem.GetDistanceMatrix(mol)
    dm[dm == 1e+8] = -1
    diameter = int(np.max(dm))
    return diameter


def get_rings(name):
    results = pcp.get_compounds(name, 'name')[0].isomeric_smiles
    mol = Chem.MolFromSmiles(results)
    rings = rdMolDescriptors.CalcNumRings(mol)
    return rings


def main():
    diff = pd.read_csv(diff_path)
    improvs = []
    for i in tqdm(range(377)):
        d1 = diff.iloc[i]['anchor_names']
        d2 = diff.iloc[i]['library_names']
        improvs.append(diff.iloc[i]['AUPRC_diff'])

    if os.path.exists('cache/diameters.npy'):
        diameters = np.load('cache/diameters.npy')
    else:
        diameters = []
        for i in tqdm(range(377)):
            d1 = diff.iloc[i]['anchor_names']
            d2 = diff.iloc[i]['library_names']
            diameters.append(get_diameter(d1) + get_diameter(d2))
        diameters = np.asarray(diameters)
        np.save('cache/diameters.npy', diameters)

    if os.path.exists('cache/numrings.npy'):
        numrings = np.load('cache/numrings.npy')
    else:
        numrings = []
        for i in tqdm(range(377)):
            d1 = diff.iloc[i]['anchor_names']
            d2 = diff.iloc[i]['library_names']
            numrings.append(get_rings(d1) + get_rings(d2))
        numrings = np.asarray(numrings)
        np.save('cache/numrings.npy', numrings)

    pval1 = stats.ttest_ind(diameters[0: 50], diameters[50: ], alternative='greater')
    pval2 = stats.ttest_ind(numrings[0: 50], numrings[50:], alternative='greater')
    print('Hypothethis top graph diameter is greater p value: {}'.format(pval1))
    print('Hypothethis top number of rings is greater p value: {}'.format(pval2))

    w = 0.6

    plt.clf()
    palette1 = ["#7fc97f", "#beaed4", "#fdc086", "#ffff99", "#386cb0"]
    group, x_jitter = [], []
    g1, g2 = [], []
    for i in range(len(diameters)):
        if diameters[i] <= 44:
            g1.append(improvs[i])
            group.append('0-44')
        if 44 < diameters[i]:
            g2.append(improvs[i])
            group.append('>=45')
    pval = stats.ttest_ind(g2, g1, alternative='greater')
    order = ['0-44', '>=45']
    sns.set_palette(palette=palette1)
    fig, ax = plt.subplots(figsize=(1*FIG_WIDTH, FIG_HEIGHT))
    ax = sns.violinplot(x=group, y=improvs, order=order)
    #plt.scatter(x_jitter, diameters, c='k', s=1.2, alpha=0.5)
    ax.set_xlabel('Graph Diameter')
    ax.set_ylabel('AUPRC Improvements')
    #ax.set_yticklabels(labels=['Top 50 Drug Combo \n With Improvements', 'Others'], fontsize=MEDIUM_SIZE)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tight_layout()
    plt.rcParams['savefig.dpi'] = 800
    plt.savefig('figure/graph_diameter.pdf')

    plt.clf()
    palette1 = ["#7fc97f", "#beaed4", "#fdc086", "#ffff99", "#386cb0"]
    order = ['0-8', '>8']
    g1, g2 = [], []
    group, x_jitter = [], []
    for i in range(len(numrings)):
        if numrings[i] <= 8:
            g1.append(improvs[i])
            group.append('0-8')
        if 8 < numrings[i] <= 16:
            g2.append(improvs[i])
            group.append('>8')
    pval = stats.ttest_ind(g2, g1, alternative='greater')
    sns.set_palette(palette=palette1)
    fig, ax = plt.subplots(figsize=(1*FIG_WIDTH, FIG_HEIGHT))
    ax = sns.violinplot(x=group, y=improvs, order=order)
    # plt.scatter(x_jitter, diameters, c='k', s=1.2, alpha=0.5)
    ax.set_xlabel('Number of Rings')
    ax.set_ylabel('AUPRC Improvements')
    #ax.set_yticklabels(labels=['Top 50 Drug Combo \n With Improvements', 'Others'], fontsize=SMALLER_SIZE)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tight_layout()
    plt.rcParams['savefig.dpi'] = 800
    #plt.show()
    plt.savefig('figure/number_of_rings.pdf')


if __name__ == '__main__':
    main()