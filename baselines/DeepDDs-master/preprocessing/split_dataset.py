from cgi import test
import pandas as pd
import random
import os
import pdb

df_ddses = pd.read_csv('baselines/DeepDDs-master/data_ours/ddses_new.csv', index_col=0)
savedir = 'baselines/DeepDDs-master/data_ours_3fold/'

os.makedirs(savedir, exist_ok=True)

length = len(df_ddses)
fold_num = 3
pot = int(length / fold_num)

print('training size', length - pot)
print('validation size', pot)

random_num = random.sample(range(0, length), length)

for i in range(fold_num):
    test_num = random_num[pot * i:pot * (i+1)]
    train_num = random_num[:pot * i] + random_num[pot * (i+1):]
    
    data_train = df_ddses.iloc[train_num, :]
    data_test = df_ddses.iloc[test_num, :]

    data_train.to_csv(os.path.join(savedir, f'train_fold{i}.csv'), index=False)
    data_test.to_csv(os.path.join(savedir, f'test_fold{i}.csv'), index=False)
