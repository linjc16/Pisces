import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler

from const import DRUG2ID_FILE, FP_FILE, DESC_FILE, DRUG_FEAT_FILE, SCALER_FILE

# drug2id
drug2id = dict()
with open(DRUG2ID_FILE, 'r') as f:
    f.readline()
    for line in f:
        d, i = line.rstrip().split('\t')
        drug2id[d] = int(i)

# fingerprints
fp_df = pd.read_csv(FP_FILE, sep='\t', usecols=['drug', 'fingerprint'], dtype=str)
fp_df = fp_df.sort_values(by='drug')
nbits = len(fp_df['fingerprint'][0])
drug_feat_fp = np.zeros((len(drug2id), nbits))
for _, (drug, fp) in fp_df.iterrows():
    for j, b in enumerate(fp):
        drug_feat_fp[drug2id[drug], j] = float(b)
col_std = np.std(drug_feat_fp, axis=0)
col_nz = col_std != 0
drug_feat_fp = drug_feat_fp[:, col_nz]

# descriptors
desc_df = pd.read_csv(DESC_FILE, sep='\t')
valid_cols = set()
for col in desc_df.columns:
    if col == 'drug':
        continue
    std = desc_df[col].var()
    if std != 0:
        valid_cols.add(col)
desc2id = dict()
for i, d in enumerate(sorted(valid_cols)):
    desc2id[d] = i
drug_feat_desc = np.zeros((len(drug2id), len(valid_cols)))
for _, row in desc_df.iterrows():
    for d, j in desc2id.items():
        drug_feat_desc[drug2id[row['drug']], j] = row[d]

# concat
feat = np.concatenate([drug_feat_fp, drug_feat_desc], axis=1)
scaler = StandardScaler().fit(feat)
feat = scaler.transform(feat)
# joblib.dump(scaler, SCALER_FILE)
# print("Saved", SCALER_FILE)
np.save(DRUG_FEAT_FILE, feat)
print("Saved", DRUG_FEAT_FILE)
