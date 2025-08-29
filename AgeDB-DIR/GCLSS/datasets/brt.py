"""EchoNet-Dynamic Dataset."""

import os
import collections
import pandas
import datetime
import cv2
import torch

import numpy as np
import skimage.draw
import torchvision
# import echonet

from scipy.special import expit
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing

class Echo(torchvision.datasets.VisionDataset):
    

    def __init__(self, root=None,
                 split="train", target_type="ground_truth",
                 mean=0., std=1.,
                 length=16, period=2,
                 max_length=250,
                 clips=1,
                 pad=None,
                 ssl_type = 0,
                 ssl_postfix = "",
                 ssl_mult = 1,
                 ssl_edesonly = True,
                 noise=None,
                 target_transform=None,
                 exclude_split = True
                 ):
        if root is None:
            assert 1==2, "need root value"

        super().__init__(root, target_transform=target_transform)

        self.split = split.upper()
        if not isinstance(target_type, list):
            target_type = [target_type]
        self.target_type = target_type
        self.mean = mean
        self.std = std
        self.length = length
        self.max_length = max_length
        self.period = period
        self.clips = clips
        self.pad = pad
        self.noise = noise
        self.target_transform = target_transform

        self.ssl_type = ssl_type
        self.ssl_postfix = ssl_postfix
        self.ssl_mult = ssl_mult

        self.ssl_edesonly = ssl_edesonly
        self.exclude_split = exclude_split

        self.fnames, self.outcome = [], []

            # Load video-level labels
        print("Using data file from ", os.path.join(self.root, "FileList{}.csv".format(self.ssl_postfix)))

        with open(os.path.join(self.root, "FileList{}.csv".format(self.ssl_postfix))) as f:
            data = pandas.read_csv(f)
        data["SPLIT"].map(lambda x: x.upper())

        if len(self.ssl_postfix) > 0:
            data_train_lab = data[(data["SPLIT"] == "TRAIN") & (data["SSL_SPLIT"] == "LABELED")].copy()
        else:
            data_train_lab = data[(data["SPLIT"] == "TRAIN")].copy()

        print(data.columns)
        if  len(self.ssl_postfix) >0:
            data_cont_ref = data[data["SPLIT"] == "TRAIN"].copy()
            data_cont_ref = data_cont_ref[data_cont_ref["SSL_SPLIT"] == "LABELED"]
        else:
            data_cont_ref = data.copy()


        if self.split != "ALL":
            data = data[data["SPLIT"] == self.split]

        # print(data)

        if self.ssl_type == 1:
            assert self.split == "TRAIN", "subset selection only for train"
            #### labeled training
            data = data[data["SSL_SPLIT"] == "LABELED"]
            print("Using SSL_SPLIT Labeled, total samples", len(data))
            #### need to double/multiply the dataset 
            data_columns = data.columns
            data = pandas.DataFrame(np.repeat(data.values,self.ssl_mult,axis=0))
            data.columns = data_columns
            print("data after duplicates:", len(data))

        elif self.ssl_type == 2:
            assert self.split == "TRAIN", "subset selection only for train"
            ### unlableled training
            if self.exclude_split:
                data = data[data["SSL_SPLIT"] == "UNLABELED"]
            else:
                data = data[data["SSL_SPLIT"] != "LABELED"]
            print("Using SSL_SPLIT unlabeled, total samples", len(data))
            data_columns = data.columns
            data = pandas.DataFrame(np.repeat(data.values,self.ssl_mult,axis=0))
            data.columns = data_columns
            print("data after duplicates:", len(data))

        elif self.ssl_type == 0:
            print("Using SSL_SPLIT ALL, total samples", len(data))
            pass
        else:
            assert 1==2, "invalid option for ssl_type in echonet data"

        # print(data)
        age_freq = data_cont_ref['ground_truth'].value_counts(dropna=False).rename_axis('age_bkt_key').reset_index(name='counts')
        age_freq = age_freq.sort_values(by=['age_bkt_key']).reset_index()

        age_dict = {}
        
        for key_itr_idx in range(len(age_freq['age_bkt_key'])):
            if key_itr_idx == 0:
                age_dict[age_freq['age_bkt_key'][key_itr_idx]] = age_freq['counts'][key_itr_idx]
            else:
                age_dict[age_freq['age_bkt_key'][key_itr_idx]] = age_dict[age_freq['age_bkt_key'][key_itr_idx-1]] + age_freq['counts'][key_itr_idx]
        
        for key_itr_idx in range(len(age_freq['age_bkt_key'])):
            age_dict[age_freq['age_bkt_key'][key_itr_idx]] = age_dict[age_freq['age_bkt_key'][key_itr_idx]] - age_freq['counts'][key_itr_idx]//2

        # for key_itr_idx in range(len(EF_freq['age_bkt_key'])):
        #     age_dict[age_freq['age_bkt_key'][key_itr_idx]] = age_dict[age_freq['age_bkt_key'][key_itr_idx]] / len(data_cont_ref)
        
        # print(set( data["ground_truth"]))
        # print(set(age_dict.keys()))
        # data['age_CLS'] = data["ground_truth"].apply(lambda x: age_dict[x])
        # print(set( data["ground_truth"]))

        ##### Brute force way to interpolate between labeles
        age_dict_interpolate = {}
        sorted_age_dict_keys = sorted(age_dict.keys())
        for key_itr in range(len(sorted_age_dict_keys)):
            current_key_val = sorted_age_dict_keys[key_itr]
            current_key_freq = age_dict[current_key_val]
            age_dict_interpolate[current_key_val] = age_dict[current_key_val]
            if key_itr < len(sorted_age_dict_keys) - 1:
                next_key_val = sorted_age_dict_keys[key_itr + 1]
                next_key_freq = age_dict[next_key_val]
                for interpolate_key in range(current_key_val + 1, next_key_val):
                    interpolated_value = current_key_freq + (next_key_freq - current_key_freq)/(next_key_val - current_key_val) * (interpolate_key - current_key_val)
                    age_dict_interpolate[interpolate_key] = interpolated_value
        
        
        for key_itr in range(sorted(age_dict.keys())[-1] + 1,120):
            age_dict_interpolate[key_itr] = age_dict[sorted(age_dict.keys())[-1]]
        for key_itr in range(0, sorted(age_dict.keys())[0]):
            age_dict_interpolate[key_itr] = 0
        
        print(age_dict)
        age_dict = age_dict_interpolate.copy()
        print(age_dict)
        data['age_CLS'] = data["ground_truth"].apply(lambda x: age_dict[x])
        self.label_length = len(data_cont_ref)

        
        


        self.header = data.columns.tolist()
        self.fnames = data["FileName"].tolist()

        self.outcome = data.values.tolist()

    def __getitem__(self, index):
        # Find filename of video

        # video_path = os.path.join(self.root, "Beijing-BRT", self.fnames[index])
        video_path = os.path.join(self.root, "AgeDB", self.fnames[index])
        # print(video_path)

        # print(video_path)
        # exit()

        # Load video into np.array
        video = cv2.imread(video_path).astype(np.float32)
        # print(video.shape)
        video = cv2.resize(video, (224, 224))
        # print(video.shape)
        # exit()
        video = video.transpose((2, 0, 1))


        # Apply normalization
        if isinstance(self.mean, (float, int)):
            video -= self.mean
        else:
            video -= self.mean.reshape(3, 1, 1)

        if isinstance(self.std, (float, int)):
            video /= self.std
        else:
            video /= self.std.reshape(3, 1, 1)

        # Set number of frames
        c, h, w = video.shape
        
        video1 = video.copy()
        video2 = video.copy()
        # if self.split == "train":

        # Gather targets
        target = []
        target_cls = []
        for t in self.target_type:
            key = self.fnames[index]
            if t == "Filename":
                target.append(self.fnames[index])
            else:
                target.append(np.float32(self.outcome[index][self.header.index(t)]))
                target_cls.append(np.float32(self.outcome[index][self.header.index('age_CLS')]))

        if target != []:
            target = tuple(target) if len(target) > 1 else target[0]
        
        if target_cls !=[]:
            target_cls = tuple(target_cls) if len(target_cls) > 1 else target_cls[0]

        if self.pad is not None and self.split == "TRAIN":

            c, h, w = video.shape

            temp1 = np.zeros((c, h + 2 * self.pad, w + 2 * self.pad), dtype=video.dtype)
            temp1[:, self.pad:-self.pad, self.pad:-self.pad] = video1
            i1, j1 = np.random.randint(0, 2 * self.pad, 2)
            video1 = temp1[:, i1:(i1 + h), j1:(j1 + w)]


            temp2 = np.zeros((c, h + 2 * self.pad, w + 2 * self.pad), dtype=video.dtype)
            temp2[:, self.pad:-self.pad, self.pad:-self.pad] = video2
            i2 = i1
            j2 = j1
            video2 = temp2[:, i2:(i2 + h), j2:(j2 + w)]      

        else:
            i1 = 0
            j1 = 0
        # return video, target

        if self.split == "TRAIN":
            if np.random.randint(0,2) == 0:
                video1 = video1[:,:,::-1].copy()
            
            if np.random.randint(0,2) == 0:
                video2 = video2[:,:,::-1].copy()


        return video1, video2, target, target_cls


    def __len__(self):
        return len(self.fnames)

    def extra_repr(self) -> str:
        """Additional information to add at end of __repr__."""
        lines = ["Target type: {target_type}", "SPLIT: {split}"]
        return '\n'.join(lines).format(**self.__dict__)

    def get_labeled_length(self):
        return self.label_length






def _defaultdict_of_lists():
    """Returns a defaultdict of lists.

    This is used to avoid issues with Windows (if this function is anonymous,
    the Echo dataset cannot be used in a dataloader).
    """

    return collections.defaultdict(list)



if __name__=='__main__':
    kwargs = {"target_type": ['ground_truth'],
              "mean": 97.84989, # [97.84989, 105.90563, 123.069916],
              "std": 66.31386, # [66.31386, 67.057724, 71.412025],
              "length": 32,
              "period": 2,
              }
    dataset_trainsub = {}
    dataset_trainsub['lb'] = Echo(root='/home/eecewang/Code/DR-project/CLSS-main/age_estimation/AgeDB_DIR', split="train", **kwargs, pad=12, ssl_postfix="_ssl_{}_{}".format(488, 0), ssl_type = 1, ssl_mult = 6, exclude_split = False)
    device = torch.device("cuda")
    phase = 'train'
    dataloader_lb = torch.utils.data.DataLoader(
                    dataset_trainsub['lb'], batch_size=32, num_workers=4, shuffle=True, pin_memory=(device.type == "cuda"), drop_last=(phase == "train"))#, worker_init_fn=worker_init_fn)
    for num_iter, (a,b,c,d) in enumerate(dataloader_lb):
        print(c, d)

