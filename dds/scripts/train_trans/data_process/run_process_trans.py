import re
from tqdm import tqdm
from rdkit import Chem
import pandas as pd
import os
import pickle as pkl
import numpy as np
import argparse
import pdb

def smi_tokenizer(smi):
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    try:
        assert re.sub('\s+', '', smi) == ''.join(tokens)
    except:
        return ''
    return ' '.join(tokens)

def clean_smiles(smiles):
    t = re.sub(':\d*', '', smiles)
    return t

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--raw_data_dir', type=str, default='data')
    args = parser.parse_args()

    print("processing start !")

    raw_data_dir = args.raw_data_dir
    output_data_dir = os.path.abspath(args.output_dir)  # ✅ change here
    os.makedirs(output_data_dir, exist_ok=True)

    drug_smiles_dir = os.path.join(raw_data_dir, 'drug_smiles.csv')
    cell_idxes_dir = os.path.join(raw_data_dir, 'cell_tpm.csv')

    cell_names = pd.read_csv(cell_idxes_dir, index_col=0)['cell_line_names']
    CELL_TO_INDEX_DICT = {cell_names[idx]: idx for idx in range(len(cell_names))}

    all_drug_dict = {}
    df_drug_smiles = pd.read_csv(drug_smiles_dir, index_col=0)
    for idx, smiles in zip(tqdm(df_drug_smiles['drug_names']), df_drug_smiles['smiles']):
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(clean_smiles(smiles)))
        all_drug_dict[idx] = smi_tokenizer(smiles)

    # CSVs
    train_csv_path = os.path.join(output_data_dir, 'train.csv')
    valid_csv_path = os.path.join(output_data_dir, 'valid.csv')
    test_csv_path = os.path.join(output_data_dir, 'test.csv')

    # Output text files
    paths = {
        "train": { "a": "train.a", "b": "train.b", "cell": "train.cell", "label": "train.label" },
        "valid": { "a": "valid.a", "b": "valid.b", "cell": "valid.cell", "label": "valid.label" },
        "test":  { "a": "test.a",  "b": "test.b",  "cell": "test.cell",  "label": "test.label" },
    }

    file_handles = {}
    for split, files in paths.items():
        file_handles[split] = {}
        for key, fname in files.items():
            file_handles[split][key] = open(os.path.join(output_data_dir, fname), 'w')

    label_dict = {}
    cell_dict = {str(i): i for i in range(len(cell_names))}
    drug_unusal = set()

    for split, df_path in [("train", train_csv_path), ("valid", valid_csv_path), ("test", test_csv_path)]:
        df = pd.read_csv(df_path)
        for a, b, cell, y in zip(tqdm(df['anchor_names']), df['library_names'], df['cell_line_names'], df['labels']):
            if a in all_drug_dict and b in all_drug_dict:
                file_handles[split]['a'].write(all_drug_dict[a] + '\n')
                file_handles[split]['b'].write(all_drug_dict[b] + '\n')
                file_handles[split]['cell'].write(str(CELL_TO_INDEX_DICT[cell]) + '\n')
                file_handles[split]['label'].write(str(y) + '\n')
                label_dict.setdefault(str(y), len(label_dict))
            else:
                if a not in all_drug_dict: drug_unusal.add(a)
                if b not in all_drug_dict: drug_unusal.add(b)

    for split in file_handles:
        for f in file_handles[split].values():
            f.close()

    with open(os.path.join(output_data_dir, 'label.dict'), 'w') as f:
        for k, v in label_dict.items():
            f.write(f"{k} {v}\n")

    with open(os.path.join(output_data_dir, 'cell.dict'), 'w') as f:
        for k, v in cell_dict.items():
            f.write(f"{k} {v}\n")

    print(f'drug_unusal: {drug_unusal}')
    print("processing done !")

if __name__ == "__main__":
    main()
