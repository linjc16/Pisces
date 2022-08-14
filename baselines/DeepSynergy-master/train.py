import random
import torch
import torch.nn.functional as F
import torch.nn as nn
from models.fnn import DeepSynergy
import os
import numpy as np
from torch.utils.data import DataLoader
from utils_test import ci, spearman, pearson, mse, rmse, save_AUCs
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.metrics import cohen_kappa_score, accuracy_score, roc_auc_score, precision_score, recall_score, balanced_accuracy_score, f1_score
from sklearn import metrics
import pandas as pd
import pdb
import argparse

os.environ["CUDA_VISIBLE_DEVICES"]="7"

# training function at each epoch
def train(model, device, data_loader_train, optimizer, epoch):
    print('Training.')
    model.train()
    # train_loader = np.array(train_loader)
    for batch_idx, data_ori in enumerate(data_loader_train):
        data = data_ori[:, :-1].to(device)
        drug1, drug2, cell = data[:, 0:256 + 346 + 200], data[:, (256 + 346 + 200):(256 + 346 + 200)*2], data[:, (256 + 346 + 200) * 2:]
        assert drug1.size(1) == drug2.size(1)
        # pdb.set_trace()
        assert cell.size(1) == 37261
        data_swap = torch.cat([drug2, drug1, cell], dim=1)
        data = torch.cat([data, data_swap], dim=0)
        y = data_ori[:, -1].view(-1, 1).long().to(device)
        y = y.squeeze(1)
        y = torch.cat([y, y])

        optimizer.zero_grad()
        output = model(data)
        # pdb.set_trace()
        loss = loss_fn(output, y)
        # print('loss', loss)
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx,
                                                                           len(drug_loader_train),
                                                                           100. * batch_idx / len(drug_loader_train),
                                                                           loss.item()))

def predicting(model, device, drug_loader_test):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_prelabels = torch.Tensor()
    print('Make prediction')
    with torch.no_grad():
        for data_ori in drug_loader_test:
            data = data_ori[:, :-1].to(device)
            y = data_ori[:, -1].view(-1, 1).long().to(device)

            output = model(data)
            # pdb.set_trace()
            ys = F.softmax(output, 1).to('cpu').data.numpy()
            predicted_labels = list(map(lambda x: np.argmax(x), ys))
            predicted_scores = list(map(lambda x: x[1], ys))
            total_preds = torch.cat((total_preds, torch.Tensor(predicted_scores)), 0)
            total_prelabels = torch.cat((total_prelabels, torch.Tensor(predicted_labels)), 0)
            total_labels = torch.cat((total_labels, y.view(-1, 1).cpu()), 0)
        
    return total_labels.numpy().flatten(), total_preds.numpy().flatten(), total_prelabels.numpy().flatten()

modeling = DeepSynergy

parser = argparse.ArgumentParser()
parser.add_argument('--fold', type=int)
parser.add_argument('--lr', type=float, default=1e-5)
args = parser.parse_args()
i = args.fold

TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 64
LR = args.lr
LOG_INTERVAL = 20
NUM_EPOCHS = 500

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)

# CPU or GPU

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('The code uses GPU...')
else:
    device = torch.device('cpu')
    print('The code uses CPU!!!')


# data_dir = '/data/linjc/dds/baselines/DeepSynergy/data_leave_cell/'
data_dir = '/data/linjc/dds/baselines/DeepSynergy/data_leave_comb/'



print(f'Run fold {i}.')
datafile_train = f'train_fold{i}'
datafile_test = f'test_fold{i}'

print(f'Loading training data from {data_dir}')
drug1_data_train = torch.load(os.path.join(data_dir, f'{datafile_train}_drug1.pt'))
drug2_data_train = torch.load(os.path.join(data_dir, f'{datafile_train}_drug2.pt'))
cell_data_train = torch.load(os.path.join(data_dir, f'{datafile_train}_cell.pt'))
drug_label_train = torch.load(os.path.join(data_dir, f'{datafile_train}_label.pt')).view(-1, 1)
# pdb.set_trace()
drug1_data_train[:, 0:200] = F.normalize(drug1_data_train[:, 0:200], dim=0)
drug2_data_train[:, 0:200] = F.normalize(drug2_data_train[:, 0:200], dim=0)
# pdb.set_trace()

feats_train = torch.cat([drug1_data_train, drug2_data_train, cell_data_train, drug_label_train], dim=1)
drug_data_train = feats_train

print(f'Loading test data from {data_dir}')
drug1_data_test = torch.load(os.path.join(data_dir, f'{datafile_test}_drug1.pt'))
drug2_data_test = torch.load(os.path.join(data_dir, f'{datafile_test}_drug2.pt'))
cell_data_test = torch.load(os.path.join(data_dir, f'{datafile_test}_cell.pt'))
drug_label_test = torch.load(os.path.join(data_dir, f'{datafile_test}_label.pt')).view(-1, 1)

drug1_data_test[:, 0:200] = F.normalize(drug1_data_test[:, 0:200], dim=0)
drug2_data_test[:, 0:200] = F.normalize(drug2_data_test[:, 0:200], dim=0)

feats_test = torch.cat([drug1_data_test, drug2_data_test, cell_data_test, drug_label_test], dim=1)
drug_data_test = feats_test

drug_loader_train = DataLoader(drug_data_train, batch_size=TRAIN_BATCH_SIZE, shuffle=None, num_workers=4)
drug_loader_test = DataLoader(drug_data_test, batch_size=TRAIN_BATCH_SIZE, shuffle=None, num_workers=4)


model = modeling().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

savedir = os.path.join(data_dir, 'results')
os.makedirs(savedir, exist_ok=True)

model_file_name = os.path.join(savedir, 'DeepSynergy(DrugA_DrugB)' + str(i) + '--model_' + datafile_train +  '.model')
result_file_name = os.path.join(savedir, 'DeepSynergy(DrugA_DrugB)' + str(i) + '--result_' + datafile_train +  '.csv')
file_AUCs = os.path.join(savedir, 'DeepSynergy(DrugA_DrugB)' + str(i) + '--AUCs--' + datafile_train + '.txt')
file_AUCs_best = os.path.join(savedir, 'DeepSynergy(DrugA_DrugB)' + str(i) + '--AUCs_best--' + datafile_train + '.txt')

AUCs = ('Epoch\tACC\tBACC\tAUC_dev\tPR_AUC\tPREC\tRECALL\tF1\tTPR\tKAPPA')
with open(file_AUCs, 'w') as f:
    f.write(AUCs + '\n')

with open(file_AUCs_best, 'w') as f:
    f.write(AUCs + '\n')

best_auc = 0
for epoch in range(NUM_EPOCHS):
    train(model, device, drug_loader_train, optimizer, epoch + 1)
    T, S, Y = predicting(model, device, drug_loader_test)
    # T is correct label
    # S is predict score
    # Y is predict label
    # pdb.set_trace()
    # compute preformence
    AUC = roc_auc_score(T, S)
    precision, recall, threshold = metrics.precision_recall_curve(T, S)
    PR_AUC = metrics.auc(recall, precision)
    BACC = balanced_accuracy_score(T, Y)
    tn, fp, fn, tp = confusion_matrix(T, Y).ravel()
    TPR = tp / (tp + fn)
    PREC = precision_score(T, Y)
    ACC = accuracy_score(T, Y)
    KAPPA = cohen_kappa_score(T, Y)
    recall = recall_score(T, Y)
    F1 = f1_score(T, Y)

    # save data
    AUCs = [epoch, ACC, BACC, AUC, PR_AUC, PREC, recall, F1, TPR, KAPPA]
    save_AUCs(AUCs, file_AUCs)
    print(f'AUC: {AUC}, PR_AUC: {PR_AUC}, ACC: {ACC}, BACC: {BACC}, \
            PREC: {PREC}, TPR: {TPR}, KAPPA: {KAPPA}, F1: {F1}, RECALL: {recall}.')
    # ret = [rmse(T, S), mse(T, S), pearson(T, S), spearman(T, S), ci(T, S)]
    if best_auc < AUC:
        best_auc = AUC
        print(best_auc)
        save_AUCs(AUCs, file_AUCs_best)
        # torch.save(model.state_dict(), model_file_name)
        # independent_num = []
        # independent_num.append(test_num)
        # independent_num.append(T)
        # independent_num.append(Y)
        # independent_num.append(S)
        # txtDF = pd.DataFrame(data=independent_num)
        # txtDF.to_csv(result_file_name, index=False, header=False)
