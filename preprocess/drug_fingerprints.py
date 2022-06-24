import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

import pdb

df_smiles = pd.read_csv('data/drug_smiles.csv')
smiles_list = df_smiles['smiles'].tolist()

drug_names = df_smiles['drug_names'].tolist()


radius=2
nBits=256

ecfp6_name = [f'Bit_{i}' for i in range(nBits)]

df_drug_fpt = []

for idx, smiles in enumerate(smiles_list):
    mol = Chem.MolFromSmiles(smiles)
    ECFP6 = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
    ecfp6_bits = list(ECFP6)
    df_drug_fpt.append(ecfp6_bits)

df_morgan = pd.DataFrame(df_drug_fpt, index = drug_names, columns=ecfp6_name)

df_morgan.to_csv('data/drug_fingerprints.csv')

