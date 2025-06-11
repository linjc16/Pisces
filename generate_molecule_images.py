import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw

# Paths
csv_path = "data/drug_smiles.csv"
output_dir = "molecule_images"
os.makedirs(output_dir, exist_ok=True)

# Load CSV
df = pd.read_csv(csv_path)

# Handle both possible column name styles
drug_col = "drug_names" if "drug_names" in df.columns else df.columns[0]
smiles_col = "smiles" if "smiles" in df.columns else df.columns[1]

# Generate images
for _, row in df.iterrows():
    name = row[drug_col].replace("/", "_").replace("\\", "_")
    smiles = row[smiles_col]
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        img_path = os.path.join(output_dir, f"{name}.png")
        Draw.MolToFile(mol, img_path)
