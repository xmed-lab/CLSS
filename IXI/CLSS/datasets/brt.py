"""EchoNet-Dynamic Dataset."""

import os
import collections
import pandas
import datetime
import cv2

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
            # print(data[target_type])
            data[target_type] = data[target_type].apply(lambda x: round(x, 1))
            # print(data[target_type])
            # exit()
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

        

        self.header = data.columns.tolist()
        self.fnames = data["FileName"].tolist()

        self.outcome = data.values.tolist()

    def __getitem__(self, index):
        # Find filename of video

        # video_path = os.path.join(self.root, "Beijing-BRT", self.fnames[index])
        video_path = os.path.join(self.root, self.fnames[index])
 
        video = np.load(video_path).astype(np.float32)
        # print(video.shape)
        video = np.expand_dims(video, axis=-1)
        # exit()
        video = video.transpose((3, 0, 1, 2))


        # Apply normalization
        if isinstance(self.mean, (float, int)):
            video -= self.mean
        else:
            # video -= self.mean.reshape(3, 1, 1, 1)
            video -= self.mean.reshape(1, 1, 1, 1)

        if isinstance(self.std, (float, int)):
            video /= self.std
        else:
            # video /= self.std.reshape(3, 1, 1, 1)
            video /= self.std.reshape(1, 1, 1, 1)

        # Set number of frames
        c, h, w, d = video.shape
        
        video1 = video.copy()
        # video2 = video.copy()
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
                # target_cls.append(np.float32(self.outcome[index][self.header.index('age_CLS')]))

        if target != []:
            target = tuple(target) if len(target) > 1 else target[0]
        

        if self.pad is not None and self.split == "TRAIN":

            c, h, w, d = video.shape

            temp1 = np.zeros((c, h + 2 * self.pad, w + 2 * self.pad, d + 2 * self.pad), dtype=video.dtype)
            temp1[:, self.pad:-self.pad, self.pad:-self.pad, self.pad:-self.pad] = video1
            i1, j1, k1 = np.random.randint(0, 2 * self.pad, 3)
            video1 = temp1[:, i1:(i1 + h), j1:(j1 + w), k1:(k1 + d)]


            # temp2 = np.zeros((c, h + 2 * self.pad, w + 2 * self.pad, d + 2 * self.pad), dtype=video.dtype)
            # temp2[:, self.pad:-self.pad, self.pad:-self.pad, self.pad:-self.pad] = video2
            # i2 = i1
            # j2 = j1
            # k2 = k1
            # video2 = temp2[:, i2:(i2 + h), j2:(j2 + w), k2:(k2 + d)]      

        else:
            i1 = 0
            j1 = 0
            k1 = 0
        # return video, target

        if self.split == "TRAIN":
            if np.random.randint(0,2) == 0:
                video1 = video1[:,:,:,::-1].copy()
            if np.random.randint(0,2) == 0:
                video1 = video1[:,:,::-1,:].copy()
            if np.random.randint(0,2) == 0:
                video1 = video1[:,::-1,:,:].copy()
            
            # if np.random.randint(0,2) == 0:
            #     video2 = video2[:,:,::-1].copy()


        return video1, video1, target


    def __len__(self):
        return len(self.fnames)

    def extra_repr(self) -> str:
        """Additional information to add at end of __repr__."""
        lines = ["Target type: {target_type}", "SPLIT: {split}"]
        return '\n'.join(lines).format(**self.__dict__)






def _defaultdict_of_lists():
    """Returns a defaultdict of lists.

    This is used to avoid issues with Windows (if this function is anonymous,
    the Echo dataset cannot be used in a dataloader).
    """

    return collections.defaultdict(list)





