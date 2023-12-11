import pandas as pd

import nibabel as nib
import os
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
import scipy.ndimage
import random
import glob
#https://github.com/bijiuni/brain_age/tree/master

RAW_PATH = "/home/wdaiaj/projects/datasources/IXI_brain_root/ixi_labels_raw.csv"
OUT_PATH = "/home/wdaiaj/projects/datasources/IXI_brain_root/IXI_brain_npy"
OUT_ROOT = "/home/wdaiaj/projects/datasources/IXI_brain_root/"
TEST_SPLIT_AMOUNT = 88
VAL_SPLIT_AMOUNT = 80

NZ_TEMPLATE = "/home/wdaiaj/projects/datasources/IXI_brain_root/IXI_brain/IXI{}*.nii.gz"

os.makedirs(OUT_PATH, exist_ok = True)



def loadData(sub_ID):

    training_img = nib.load(sub_ID)

    training_pixdim = training_img.header['pixdim'][1:4]
    
    training_data = training_img.get_fdata()
    
    return training_data, training_pixdim


def resample(image, pixdim, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    spacing = pixdim

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    
    return image




def preprocess(input_path, save_path_file):
    
    single_data, single_dim = loadData(input_path)

    single_resampled = resample(single_data, single_dim, [2,2,2])

    DESIRED_SHAPE = (130, 130, 110)
    X_before = int((DESIRED_SHAPE[0]-single_resampled.shape[0])/2)
    Y_before = int((DESIRED_SHAPE[1]-single_resampled.shape[1])/2)
    Z_before = int((DESIRED_SHAPE[2]-single_resampled.shape[2])/2)

    npad = ((X_before, DESIRED_SHAPE[0]-single_resampled.shape[0]-X_before), (Y_before, DESIRED_SHAPE[1]-single_resampled.shape[1]-Y_before), (Z_before, DESIRED_SHAPE[2]-single_resampled.shape[2]-Z_before))
    single_padded = np.pad(single_resampled, pad_width=npad, mode='constant', constant_values=0)

    np.save(os.path.join(OUT_ROOT, save_path_file), single_padded)






raw_data_in = pd.read_csv(RAW_PATH)
raw_data_in['orig_path'] = "MISSING"

id_list = list(raw_data_in['IXI_ID'])

for id_itr in id_list:
    id_itr_zfill = str(id_itr).zfill(3)
    orig_path = NZ_TEMPLATE.format(id_itr_zfill)
    found_matching_path = glob.glob(orig_path)

    if len(found_matching_path) == 1:
        raw_data_in.loc[raw_data_in['IXI_ID'] == id_itr, 'orig_path'] = found_matching_path[0]
    elif len(found_matching_path)> 1:
        assert False, "multimatch"



raw_data_in = raw_data_in[~(raw_data_in['orig_path'] == 'MISSING')]

raw_data_in['FileName'] = "IXI_brain_npy/" + raw_data_in['IXI_ID'].astype(str).str.zfill(3) + ".npy"
raw_data_in['SPLIT'] = "TRAIN"
raw_data_in['IDX'] = raw_data_in.index

index_list = [i for i in range(len(raw_data_in))]

random.seed(0)
random.shuffle(index_list)



index_list_test = index_list[:TEST_SPLIT_AMOUNT]
index_list_val = index_list[TEST_SPLIT_AMOUNT : TEST_SPLIT_AMOUNT + VAL_SPLIT_AMOUNT]


raw_data_in.loc[raw_data_in['IDX'].isin(index_list_test), 'SPLIT'] = 'TEST'
raw_data_in.loc[raw_data_in['IDX'].isin(index_list_val), 'SPLIT'] = 'VAL'

raw_data_in.reset_index()

print(raw_data_in)


for intr_idx in range(len(raw_data_in)):
    if intr_idx % 5 == 0:
        print("processing {} out of {}".format(intr_idx, len(raw_data_in)))
    raw_data_itr = raw_data_in.iloc[intr_idx]
    raw_data_itr_path = raw_data_itr['orig_path']
    raw_data_itr_path_out = raw_data_itr['FileName']

    preprocess(raw_data_itr_path, raw_data_itr_path_out)

    # exit()

raw_data_in.to_csv("/home/wdaiaj/projects/datasources/IXI_brain_root/FileList.csv", index = False)
quit()








