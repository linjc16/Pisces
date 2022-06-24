from math import isnan
import pandas as pd
import pdb
import re

df_scr = pd.read_csv('data_feat/screened_compunds_rel_8.2.csv') # download from https://www.cancerrxgene.org/downloads/bulk_download
df_smiles = pd.read_csv('data/drug_smiles.csv')

targets = df_scr['TARGET']
drugs = df_smiles['drug_names']

target_set = set()

for i in range(len(targets)):
    if not isinstance(targets[i], str) and isnan(targets[i]):
        continue
    targets_curr = targets[i].split(',')
    for target in targets_curr:
        target_set.add(target.replace(' ', ''))

print(f'num of targets: {len(target_set)}')
target_list = list(sorted(target_set))

TARGET_TO_INDEX_DICT = {target: idx for idx, target in enumerate(target_list)}

drug_target_dict = {'drug_names': drugs.tolist()}

for i in range(len(target_list)):
    drug_target_dict[target_list[i]] = [0] * len(drugs)

drug_target_df = pd.DataFrame.from_dict(drug_target_dict)

drug_protein_pairs = []

for i in range(len(drugs)):
    target_temp = targets[df_scr['DRUG_NAME'] == drugs[i]].tolist()
    # pdb.set_trace()
    if len(target_temp) == 0:
        print(drugs[i])
        continue
    target_temp = target_temp[0].split(',')
    for target in target_temp:
        target = target.replace(' ', '')
        drug_protein_pairs.append([drugs[i], target])
        drug_target_df.loc[i, target] = 1

drug_target_df.copy().to_csv('data/drug_target.csv')

drug_protein_df = pd.DataFrame(drug_protein_pairs, columns=['drug', 'protein'])

drug_protein_df.to_csv('data/drug_protein.csv', index=False)
# pdb.set_trace()