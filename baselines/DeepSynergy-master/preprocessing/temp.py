import numpy as np
import torch
import os


data_dir = '/data/linjc/dds/baselines/DeepSynergy/data/'

for i in range(5):
    print(f'Run fold {i}.')
    datafile_train = f'train_fold{i}'
    datafile_test = f'test_fold{i}'

    drug_data_train = np.load(os.path.join(data_dir, f'{datafile_train}_data.npy'))
    drug_label_train = np.load(os.path.join(data_dir, f'{datafile_train}_label.npy'))
    
    drug_data_train = torch.from_numpy(drug_data_train)
    drug_label_train = torch.from_numpy(drug_label_train)

    torch.save(drug_data_train, os.path.join(data_dir, f'{datafile_train}_data.pt'))
    torch.save(drug_label_train, os.path.join(data_dir, f'{datafile_train}_label.pt'))


    drug_data_test = np.load(os.path.join(data_dir, f'{datafile_test}_data.npy'))
    drug_label_test = np.load(os.path.join(data_dir, f'{datafile_test}_label.npy'))

    drug_data_test = torch.from_numpy(drug_data_test)
    drug_label_test = torch.from_numpy(drug_label_test)

    torch.save(drug_data_test, os.path.join(data_dir, f'{datafile_test}_data.pt'))
    torch.save(drug_label_test, os.path.join(data_dir, f'{datafile_test}_label.pt'))