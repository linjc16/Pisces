import re
from tqdm import tqdm
from rdkit import Chem
import pandas as pd
import os
import pickle as pkl
import numpy as np
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
    print("processing start !")
    
    raw_data_dir = 'data'

    output_data_dir = 'data/transductive/fold1'
    os.makedirs(output_data_dir, exist_ok=True)
    
    drug_smiles_dir = os.path.join(raw_data_dir, 'drug_smiles.csv')
    
    cell_idxes = os.path.join(raw_data_dir, 'cell_features.csv')

    all_drug_dict = {}
    df_drug_smiles = pd.read_csv(drug_smiles_dir)
    for idx, smiles in zip(tqdm(df_drug_smiles['drug_names']), df_drug_smiles['smiles']):
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(clean_smiles(smiles)))
        all_drug_dict[idx] = smi_tokenizer(smiles)
    
    train_csv_pos_path = os.path.join(output_data_dir, 'train_pos.csv')
    train_csv_neg_path = os.path.join(output_data_dir, 'train_neg.csv')
    valid_csv_pos_path = os.path.join(output_data_dir, 'valid_pos.csv')
    valid_csv_neg_path = os.path.join(output_data_dir, 'valid_neg.csv')
    test_csv_pos_path = os.path.join(output_data_dir, 'test_pos.csv')
    test_csv_neg_path = os.path.join(output_data_dir, 'test_neg.csv')

    train_a_dir = os.path.join(output_data_dir, 'train.a')
    train_b_dir = os.path.join(output_data_dir, 'train.b')
    valid_a_dir = os.path.join(output_data_dir, 'valid.a')
    valid_b_dir = os.path.join(output_data_dir, 'valid.b')
    test_a_dir = os.path.join(output_data_dir, 'test.a')
    test_b_dir = os.path.join(output_data_dir, 'test.b')

    train_nega_dir = os.path.join(output_data_dir, 'train.nega')
    train_negb_dir = os.path.join(output_data_dir, 'train.negb')
    valid_nega_dir = os.path.join(output_data_dir, 'valid.nega')
    valid_negb_dir = os.path.join(output_data_dir, 'valid.negb')
    test_nega_dir = os.path.join(output_data_dir, 'test.nega')
    test_negb_dir = os.path.join(output_data_dir, 'test.negb')

    train_cell_dir = os.path.join(output_data_dir, 'train.cell')
    valid_cell_dir = os.path.join(output_data_dir, 'valid.cell')
    test_cell_dir = os.path.join(output_data_dir, 'test.cell')

    train_neg_cell_dir = os.path.join(output_data_dir, 'train.negcell')
    valid_neg_cell_dir = os.path.join(output_data_dir, 'valid.negcell')
    test_neg_cell_dir = os.path.join(output_data_dir, 'test.negcell')

    cell_dict = {}

    with open(train_a_dir, 'w') as ta_w, open(train_nega_dir, 'w') as tna_w, \
        open(train_b_dir, 'w') as tb_w, open(train_negb_dir, 'w') as tnb_w, \
        open(valid_a_dir, 'w') as va_w, open(valid_nega_dir, 'w') as vna_w, \
        open(valid_b_dir, 'w') as vb_w, open(valid_negb_dir, 'w') as vnb_w, \
        open(test_a_dir, 'w') as tsa_w, open(test_nega_dir, 'w') as tsna_w, \
        open(test_b_dir, 'w') as tsb_w, open(test_negb_dir, 'w') as tsnb_w, \
        open(train_cell_dir, 'w') as tc, open(valid_cell_dir, 'w') as vc, \
        open(test_cell_dir, 'w') as tsc, open(train_neg_cell_dir, 'w') as tnc, \
        open(valid_neg_cell_dir, 'w') as vnc, open(test_neg_cell_dir, 'w') as tsnc:

        train_csv_pos = pd.read_csv(train_csv_pos_path)
        for a, b, c in zip(tqdm(train_csv_pos['anchor_names']), train_csv_pos['library_names'], \
            train_csv_pos['cell_line_names']):
            ta_w.writelines(all_drug_dict[a] + '\n')
            tb_w.writelines(all_drug_dict[b] + '\n')
            tc.writelines(cell_idxes[c] + '\n')
            
            if str(cell_idxes[c]) not in cell_dict:
                cell_dict[str(cell_idxes[c])] = len(cell_dict)
        
        train_csv_neg = pd.read(train_csv_neg_path)
        for na, nb, nc in zip(tqdm(train_csv_neg['anchor_names']), train_csv_neg['library_names'], \
            train_csv_neg['cell_line_names']):
            tna_w.writelines(all_drug_dict[na] + '\n')
            tnb_w.writelines(all_drug_dict[nb] + '\n')
            tnc.writelines(cell_idxes[nc] + '\n')

            


    pdb.set_trace()
if __name__ == "__main__":
    main()
