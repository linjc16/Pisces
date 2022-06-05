# import pandas as pd

# drug_smiles = pd.read_csv('data/drug_smiles.csv')

# smiles = drug_smiles['smiles']

# for smile in smiles:
#     if len(smile) > 1:
#         print(smile)

import torch

logits = torch.randn([16, 1, 1])
# labels = torch.