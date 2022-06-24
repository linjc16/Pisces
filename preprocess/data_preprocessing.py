import pandas as pd
import pubchempy as pcp
import pdb
from collections import defaultdict
import os
import numpy as np



def gen_dds(combos, savedir):

    dds_set = set()
    for combo in combos:
        dds_set |= set([data for data in zip(combo['Cell Line name'], combo['Anchor Name'], combo['Library Name'], combo['Synergy?'])])

    dds_list = list(dds_set)

    # tri_dict = defaultdict(list)

    # for idx, dds in enumerate(dds_list):
    #     if dds[0:3] in tri_dict.keys():
    #         tri_dict[dds[0:3]].append((idx, dds[-1]))
    #         print(f'{dds}: {tri_dict[dds[0:3]]}')
    #     tri_dict[dds[0:3]].append((idx, dds[-1]))

    # pdb.set_trace()
    output = [item for item in map(list, zip(*dds_list))]

    dds_df = pd.DataFrame.from_dict({'cell_line_names': output[0], 'anchor_names': output[1], 'library_names': output[2], 'labels': output[3]})
    dds_df.to_csv(os.path.join(savedir, 'ddses.csv'))


def gen_drug_smiles(combos, savedir):

    drug_set = set([])

    for combo in combos:
        drug_set |= set(combo['Anchor Name'])
        drug_set |= set(combo['Library Name'])


    drug_list = list(drug_set)
    smiles_dict = defaultdict()

    for drug_name in drug_list:
        results = pcp.get_compounds(drug_name, 'name')
        for compound in results:
            smiles_dict[drug_name] = compound.isomeric_smiles
            break


    smiles_df = pd.DataFrame.from_dict({'drug_names': smiles_dict.keys(), 'smiles': smiles_dict.values()})
    smiles_df.to_csv(os.path.join(savedir, 'drug_smiles.csv'))

def gen_cell_one_hot_features(ddses_path, savedir):
    df_ddses = pd.read_csv(ddses_path)

    cell_lines = list(sorted(set(df_ddses['cell_line_names'])))

    CELL_TO_INDEX_DICT = {cell_line: idx for idx, cell_line in enumerate(cell_lines)}

    cells_df = pd.DataFrame.from_dict({'cell_line_naems': CELL_TO_INDEX_DICT.keys(), 'idx': CELL_TO_INDEX_DICT.values()})
    cells_df.to_csv(os.path.join(savedir, 'cell_features.csv'))

def gen_cell_features_expression(savedir):
    df_ddses = pd.read_csv('data/ddses.csv')
    cell_lines = list(sorted(set(df_ddses['cell_line_names'])))

    sample_info = pd.read_csv('data/raw/cell_lines/sample_info.csv')
    CCLE_expression = pd.read_csv('data/raw/cell_lines/CCLE_expression.csv')

    cell_features = []
    
    df_heads = CCLE_expression.iloc[:, 1:].keys().tolist()
    df_heads.insert(0, 'cell_line_names')

    # pdb.set_trace()
    for cell_line in cell_lines:
        
        DepMap_id = sample_info['DepMap_ID'][sample_info['cell_line_name'].str.lower() == cell_line.lower()]

        if len(DepMap_id) > 0:
            try:
                cl_feat = CCLE_expression[CCLE_expression.iloc[:, 0] == DepMap_id.iloc[0]].iloc[:, 1:].to_numpy().tolist()[0]
                cl_feat.insert(0, cell_line)
                assert len(cl_feat) == len(df_heads)
                cell_features.append(cl_feat)
            except:
                print(f'{cell_line} not in CCLE_expression.csv')
        else:
            # pdb.set_trace()
            print(f'{cell_line} not in samples_info.csv')
    
    cells_df = pd.DataFrame(cell_features, columns=df_heads)
    cells_df.to_csv(os.path.join(savedir, 'cell_features_expression.csv'))

if __name__ == '__main__':

    b_combo = pd.read_csv('data/raw/breast_anchor_combo.csv')
    c_combo = pd.read_csv('data/raw/colon_anchor_combo.csv')
    p_combo = pd.read_csv('data/raw/pancreas_anchor_combo.csv')
    combos = [b_combo, c_combo, p_combo]
    savedir = 'data'

    # pdb.set_trace()
    gen_dds(combos, savedir)
    gen_drug_smiles(combos, savedir)