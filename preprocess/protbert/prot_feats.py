from transformers import BertModel, BertTokenizer
import torch
import pandas as pd
from tqdm import tqdm
import pdb


prot_seq_df = pd.read_csv('data/protein_sequence/protein_seq_processed.csv')

prot_seq = prot_seq_df['Sequences'].tolist()

tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False)
ProtBert = BertModel.from_pretrained("Rostlab/prot_bert_bfd").cuda()

encoded_input = tokenizer(prot_seq, padding=True, truncation=True, max_length=1024, return_tensors='pt')

# pdb.set_trace()
prot_feats = []
batch_size = 128
with torch.no_grad():
    for i in tqdm(range(len(prot_seq) // batch_size + 1)):
        encoded_input_curr = {}
        encoded_input_curr['input_ids'] = encoded_input['input_ids'][i * batch_size : (i + 1) * batch_size, :].cuda()
        encoded_input_curr['token_type_ids'] = encoded_input['token_type_ids'][i * batch_size : (i + 1) * batch_size, :].cuda()
        encoded_input_curr['attention_mask'] = encoded_input['attention_mask'][i * batch_size : (i + 1) * batch_size, :].cuda()
        prot_feats.append(ProtBert(**encoded_input_curr)['pooler_output'])
    
    prot_feats = torch.cat(prot_feats, dim=0)

    torch.save(prot_feats, 'data/protein_sequence/prot_feats.pt')