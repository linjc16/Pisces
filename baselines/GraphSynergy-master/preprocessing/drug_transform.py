import pandas as pd
import pdb



aliases_df = pd.read_csv('data/raw/9606.protein.aliases.v11.5.txt', sep='\t', names=['#string_protein_id', 'alias', 'source'], header=0)
string_protein_ids = aliases_df['#string_protein_id']

drug_target_raw = pd.read_csv('data/drug_protein_raw.csv')

ensp2node_df = pd.read_csv('baselines/PRODeepSyn-main/cell/data/ensp2node.tsv', sep='\t')

ENSP_TO_NODE_DICT = {ensp: node for ensp, node in zip(ensp2node_df['ensp_id'].tolist(), ensp2node_df['node_id'].tolist())}
# pdb.set_trace()

targets = set()

for i in range(len(drug_target_raw)):
    targets.add(drug_target_raw['protein'].iloc[i])
    # pdb.set_trace()

targets = list(targets)

TARGET_TO_ENSP_DICT = {}

for target in targets:
    string_protein_id = string_protein_ids[aliases_df['alias'] == target]
    if len(string_protein_id) == 0:
        print(f'protein {target} not found.')
        continue
    TARGET_TO_ENSP_DICT[target] = string_protein_id.iloc[0]

drug_target = []

for i in range(len(drug_target_raw)):
    item = drug_target_raw.iloc[i, :]
    # pdb.set_trace()
    if item['protein'] in TARGET_TO_ENSP_DICT.keys():
        ensp_id = TARGET_TO_ENSP_DICT[item['protein']]
        if ensp_id in ENSP_TO_NODE_DICT:
            drug_target.append([item['drug'], ENSP_TO_NODE_DICT[ensp_id]])
        else:
            print(item['drug'])
            # drug_target.append([item['drug'], 17160])
    else:
        print(item['drug'])
        # drug_target.append([item['drug'], 17160])

# pdb.set_trace()

drug_target = pd.DataFrame(drug_target, columns=['drug', 'protein'])
drug_target.to_csv('baselines/GraphSynergy-master/data_ours/drug_protein.csv', index=False)