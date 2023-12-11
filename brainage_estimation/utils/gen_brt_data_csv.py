import pandas as pd
import os
import scipy.io
from os import listdir
from os.path import isfile, join
import random

def get_count(filename:str, path):
    filename = filename.split('.')[0]
    filename = filename + '.mat'
    file = os.path.join(path, filename)
    mat = scipy.io.loadmat(file)
    return len(mat['loc'])

train_path = '/home/whngak/sslreg_ageestimate/datasets/Beijing-BRT-dataset/train'

Filelist = pd.DataFrame(columns = ['index', 'FileName', 'ground_truth', 'SPLIT'])

train_jpg_path = join(train_path, 'frame')
train_gt_path = join(train_path, 'ground_truth')

frame_train = [f for f in listdir(train_jpg_path) if isfile(join(train_jpg_path, f))]
for i in range(len(frame_train)):
    Filelist = Filelist.append({'index':i, 'FileName':frame_train[i], 'ground_truth':get_count(frame_train[i], train_gt_path), 'SPLIT':'TRAIN'}, ignore_index=True)

test_path = '/home/whngak/sslreg_ageestimate/datasets/Beijing-BRT-dataset/test'
test_jpg_path = join(test_path, 'frame')
test_gt_path = join(test_path, 'ground_truth')

frame_test = [f for f in listdir(test_jpg_path) if isfile(join(test_jpg_path, f))]

for i in range(len(frame_test)):
    rand = random.random()
    if rand < 0.8:
        Filelist = Filelist.append({'index':i, 'FileName':frame_test[i], 'ground_truth':get_count(frame_test[i], test_gt_path), 'SPLIT':'TEST'}, ignore_index=True)
    else:
        Filelist = Filelist.append({'index':i, 'FileName':frame_test[i], 'ground_truth':get_count(frame_test[i], test_gt_path), 'SPLIT':'VAL'}, ignore_index=True)
print(Filelist)

Filelist.to_csv('/home/whngak/sslreg_ageestimate/datasets/Beijing-BRT-dataset/FileList.csv', index=False)