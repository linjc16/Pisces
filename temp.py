from operator import index
import pandas as pd
import pdb
import numpy as np
from zmq import device

# df_all_data = pd.read_csv('data_feat/rnaseq_all_data_20220510.csv')
# df_fpkm = pd.read_csv('data_feat/rnaseq_fpkm_20220510.csv')
# df_read_count = pd.read_csv('data_feat/rnaseq_fpkm_20220510.csv')
# df_tqm = pd.read_csv('data_feat/rnaseq_tpm_20220510.csv')

# df_cell = pd.read_csv('data/drug_descriptor.csv')

# df_cell_np = df_cell.iloc[:, 1:].to_numpy()

# tpm = pd.read_csv('data/cell_tpm.csv', index_col=0)

# feats = tpm.iloc[:, 1:].to_numpy()


# # ensp2node_df = pd.read_csv('data_feat/protein_sequence.tsv', sep='\t')

# pdb.set_trace()

# TOKENS = ['L', 'A', 'G', 'V', 'E', 'S', 'I', 'K', 'R', 'D', 'T', 'P', 'N', 'Q', 'F', 'Y', 'M', 'H', 'C', 'W', 'X', 'U', 'B', 'Z', 'O']

from transformers import BertModel, BertTokenizer
import re
tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False)
model = BertModel.from_pretrained("Rostlab/prot_bert_bfd").cuda()
# sequence_Example = "A E T C Z A O"
# sequence_Example = 'M G C G G S R A D A I E P R Y Y E S W T R E T E S T W L T Y T D S D A P P S A A A P D S G P E A G G L H S G M L E D G L P S N G V P R S T A P G G I P N P E K K T N C E T Q C P N P Q S L S S G P L T Q K Q N G L Q T T E A K R D A K R M P A K E V T I N V T D S I Q Q M D R S R R I T K N C V N'
# sequence_Example = re.sub(r"[UZOB]", "X", sequence_Example)
# prot_seq_df = pd.read_csv('data/protein_sequence/protein_seq_processed.csv')['Sequences'].tolist()

sequence_Example = ['A E T C Z A O', 'A E Z A O']
# sequence_Example = sequence_Example.replace(' ', '')

# pdb.set_trace()

encoded_input = tokenizer(sequence_Example, return_tensors='pt', padding=True, truncation=True, max_length=1024)
encoded_input['input_ids'] = encoded_input['input_ids'].cuda()
encoded_input['token_type_ids'] = encoded_input['token_type_ids'].cuda()
encoded_input['attention_mask'] = encoded_input['attention_mask'].cuda()
output = model(**encoded_input)

pdb.set_trace()

# x1 = pd.read_csv('data/protein_sequence/protein_seq_processed.csv')
# x2 = pd.read_csv('data/protein_sequence/cell_tpm_processed.csv')

# pdb.set_trace()
# assert x1['Proteins'].tolist() == x2.columns[1:].tolist()

# pdb.set_trace()
# import torch
# temp = torch.load('data/protein_sequence/prot_feats.pt', map_location='cpu')

pdb.set_trace()