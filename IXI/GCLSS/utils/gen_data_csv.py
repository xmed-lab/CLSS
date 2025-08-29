import os
import pandas as pd
import random



TARGET_TEST_SPLIT = "/home/wdaiaj/projects/datasources/UTKROOT/train_test_split/utk_test.csv"
TARGET_TRAIN_SPLIT = "/home/wdaiaj/projects/datasources/UTKROOT/train_test_split/utk_train.csv"




test_data_raw = pd.read_csv(TARGET_TEST_SPLIT)
train_data_raw = pd.read_csv(TARGET_TRAIN_SPLIT)

train_samp_num = len(train_data_raw)
train_samp_num_val = int(len(train_data_raw) * 0.2)
train_samp_num_trn = train_samp_num - train_samp_num_val

print("train_samp_num_val, train_samp_num_trn", train_samp_num_val, train_samp_num_trn)

samp_list = [i for i in range(train_samp_num)]


random.shuffle(samp_list)
# print("samp_list", samp_list)

val_indx = samp_list[:train_samp_num_val]
trn_indx = samp_list[train_samp_num_val:]

train_data_val = train_data_raw.iloc[val_indx]
train_data_trn = train_data_raw.iloc[trn_indx]

train_data_val['SPLIT'] = 'VAL'
train_data_trn['SPLIT'] = 'TRAIN'
test_data_raw['SPLIT'] = 'TEST'

data_all_val = train_data_trn.append(train_data_val)
data_all_val = data_all_val.append(test_data_raw)

data_all_val = data_all_val.reset_index()

exit()
data_all_val.to_csv("/home/wdaiaj/projects/datasources/UTKROOT/train_test_split/utk_all_wval.csv", index = False)

# print(train_data_val)
