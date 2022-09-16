import random
import torch.nn.functional as F
import torch.nn as nn
from models.gat import GATNet
from models.gat_gcn_test import GAT_GCN
from models.gcn import GCNNet
from models.ginconv import GINConvNet
from utils_test import *
from sklearn.metrics import f1_score, roc_curve, confusion_matrix
from sklearn.metrics import cohen_kappa_score, accuracy_score, roc_auc_score, precision_score, recall_score, balanced_accuracy_score
from sklearn import metrics
import pandas as pd
import pdb
import argparse


# training function at each epoch
def train(model, device, drug1_loader_train, drug2_loader_train, optimizer, epoch):
    print('Training on {} samples...'.format(len(drug1_loader_train.dataset)))
    model.train()

    for batch_idx, data in enumerate(zip(drug1_loader_train, drug2_loader_train)):
        data1 = data[0]
        data2 = data[1]
        data1 = data1.to(device)
        data2 = data2.to(device)

        y = data[0].y.view(-1, 1).long().to(device)
        y = y.squeeze(1)
        optimizer.zero_grad()
        output = model(data1, data2)
        loss = loss_fn(output, y)
        # print('loss', loss)
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx,
                                                                           len(drug1_loader_train),
                                                                           100. * batch_idx / len(drug1_loader_train),
                                                                           loss.item()))


def predicting(model, device, drug1_loader_test, drug2_loader_test):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_prelabels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(drug1_loader_test.dataset)))
    with torch.no_grad():
        for data in zip(drug1_loader_test, drug2_loader_test):
            data1 = data[0]
            data2 = data[1]
            data1 = data1.to(device)
            data2 = data2.to(device)
            output = model(data1, data2)
            ys = F.softmax(output, 1).to('cpu').data.numpy()
            predicted_labels = list(map(lambda x: np.argmax(x), ys))
            predicted_scores = list(map(lambda x: x[1], ys))
            total_preds = torch.cat((total_preds, torch.Tensor(predicted_scores)), 0)
            total_prelabels = torch.cat((total_prelabels, torch.Tensor(predicted_labels)), 0)
            total_labels = torch.cat((total_labels, data1.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten(), total_prelabels.numpy().flatten()


def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


def split_dataset(dataset, ratio):
    n = int(len(dataset) * ratio)
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2

modeling = GCNNet

TRAIN_BATCH_SIZE = 128
TEST_BATCH_SIZE = 128
LR = 0.0005
LOG_INTERVAL = 20
NUM_EPOCHS = 200

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)


# CPU or GPU

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('The code uses GPU...')
else:
    device = torch.device('cpu')
    print('The code uses CPU!!!')


# datadir = '/data/linjc/dds/baselines/DeepDDS/data_tpm'
fold_num = 1
# datadir = '/data/linjc/dds/baselines/DeepDDS/data_leave_cell'
datadir = '/data/linjc/dds/baselines/DeepDDS/data_leave_comb'
results_dir = os.path.join(datadir, 'results_')
os.makedirs(results_dir, exist_ok=True)

for i in range(fold_num):
    print(f'Run fold {i}.')
    datafile_train = f'train_fold{i}'
    datafile_test = f'test_fold{i}'

    drug1_data_train = TestbedDataset(root=datadir, dataset=datafile_train + '_drug1')
    drug1_data_test = TestbedDataset(root=datadir, dataset=datafile_test + '_drug1')

    drug1_loader_train = DataLoader(drug1_data_train, batch_size=TRAIN_BATCH_SIZE, shuffle=None, num_workers=4)
    drug1_loader_test = DataLoader(drug1_data_test, batch_size=TRAIN_BATCH_SIZE, shuffle=None, num_workers=4)

    drug2_data_train = TestbedDataset(root=datadir, dataset=datafile_train + '_drug2')
    drug2_data_test = TestbedDataset(root=datadir, dataset=datafile_test + '_drug2')
    drug2_loader_train = DataLoader(drug2_data_train, batch_size=TRAIN_BATCH_SIZE, shuffle=None, num_workers=4)
    drug2_loader_test = DataLoader(drug2_data_test, batch_size=TRAIN_BATCH_SIZE, shuffle=None, num_workers=4)
    
    # pdb.set_trace()
    model = modeling().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    
    model_file_name = os.path.join(datadir, 'results/GCNNet(DrugA_DrugB)' + str(i) + '--model_' + datafile_train +  '.model')
    result_file_name = os.path.join(datadir, 'results/GCNNet(DrugA_DrugB)' + str(i) + '--result_' + datafile_train +  '.csv')
    file_AUCs = os.path.join(datadir, 'results/GCNNet(DrugA_DrugB)' + str(i) + '--AUCs--' + datafile_train + '.txt')
    file_AUCs_best = os.path.join(datadir, 'results/GCNNet(DrugA_DrugB)' + str(i) + '--AUCs_best--' + datafile_train + '.txt')
    AUCs = ('Epoch\tACC\tBACC\tAUC_dev\tPR_AUC\tPREC\tRECALL\tF1\tTPR\tKAPPA')
    with open(file_AUCs, 'w') as f:
        f.write(AUCs + '\n')
    
    with open(file_AUCs_best, 'w') as f:
        f.write(AUCs + '\n')

    best_auc = 0
    for epoch in range(NUM_EPOCHS):
        train(model, device, drug1_loader_train, drug2_loader_train, optimizer, epoch + 1)
        T, S, Y = predicting(model, device, drug1_loader_test, drug2_loader_test)
        # T is correct label
        # S is predict score
        # Y is predict label

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
             PREC: {PREC}, TPR: {TPR}, KAPPA: {KAPPA}, RECALL: {recall}.')

        if best_auc < AUC:
            best_auc = AUC
            print(best_auc)
            save_AUCs(AUCs, file_AUCs_best)
            torch.save(model.state_dict(), model_file_name)

