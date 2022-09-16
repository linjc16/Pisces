import pandas as pd
import os
import pdb


savedir = 'baselines/PRODeepSyn-main/drug/data_ours'
# descriptors
df_drug_desc = pd.read_csv('data/drug_descriptor.csv', index_col=0)

df_drug_desc_new = {'drug': df_drug_desc.index.tolist()}

for key in df_drug_desc.keys():
    df_drug_desc_new[key] = df_drug_desc[key].replace(r'\n', ' ', regex=True).tolist()

df_drug_desc_new = pd.DataFrame.from_dict(df_drug_desc_new)

df_drug_desc_new.to_csv(os.path.join(savedir, 'descriptors.tsv'), sep='\t', index=False, header=True)

# smiles
df_drug_smiles = pd.read_csv('data/drug_smiles.csv', index_col=0)

df_drug_smiles_new = {
    'drug': df_drug_smiles['drug_names'].tolist(),
    'smiles': df_drug_smiles['smiles'].tolist()}

df_drug_smiles_new = pd.DataFrame.from_dict(df_drug_smiles_new)
df_drug_smiles_new.to_csv(os.path.join(savedir, 'smiles.csv'), index=False, header=False)

# drug2id
df_drug2id = {
    'drug': df_drug_smiles['drug_names'].tolist(),
    'id': list(range(len(df_drug_smiles)))}

df_drug2id = pd.DataFrame.from_dict(df_drug2id)
df_drug2id.to_csv(os.path.join(savedir, 'drug2id.tsv'), sep='\t', index=False, header=True)

# fp256
df_fp = pd.read_csv('data/drug_fingerprints.csv', index_col=0)
df_fp_new = {'drug': df_fp.index.tolist()}
fps = []
for i in range(len(df_fp)):
    fps.append("".join(str (x) for x in df_fp.iloc[i, 1:]))
df_fp_new['fingerprint'] = fps
# pdb.set_trace()
df_fp_new = pd.DataFrame.from_dict(df_fp_new)
df_fp_new.to_csv(os.path.join(savedir, 'fp256.tsv'), sep='\t', index=False, header=True)
# pdb.set_trace()