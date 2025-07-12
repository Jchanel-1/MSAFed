import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms as T

import os
from glob import glob
import random
import numpy as np
import json
import pdb
import pandas as pd
import pickle


class Polyp(Dataset):
    def __init__(self, client_idx=None, split='train', transform=None):
        assert split in ['train', 'test',"val"]

        self.num_classes = 2
        self.dict={
            "client1":"Kvasir",
            "client2":"ETIS",
            "client3":"CVC-ColonDB",
            "client4":"CVC-ClinicDB"
        }
        self.client_name = ['client1', 'client2', 'client3', 'client4']
        self.client_idx = client_idx    
        self.split = split
        self.transform = transform

        self.data_list = []

        with open("/data3/jjj/polyp/data/data_split/FedPolyp_npy/{}_{}.txt".format(self.client_name[int(self.client_idx)-1], split), "r") as f: 
            self.data_list = json.load(f)

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx: int):
        data_path = os.path.join('/data3/jjj/polyp/data',self.data_list[idx])
        data = np.load(data_path,allow_pickle=True)
        data=data.item()
        image = data['data'][..., 0:3]  
        label = data['data'][..., 3:]   


        sample = {'image':image, 'label':label}
        if self.transform is not None:
            sample = self.transform(sample) 

        return idx, sample


class RandomCrop_Polyp(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        GAP_H = image.shape[0] - self.output_size[0]
        GAP_W = image.shape[1] - self.output_size[1]

        while(1):
            H = random.randint(0, GAP_H)
            W = random.randint(0, GAP_W)
            if label[H:H+self.output_size[0], W:W+self.output_size[1], :].sum() > 0:
                image = image[H:H+self.output_size[0], W:W+self.output_size[1], :]
                label = label[H:H+self.output_size[0], W:W+self.output_size[1], :]
                break

        return {'image': image, 'label': label}


class ToTensor(object):
    def __call__(self, sample):
        image = sample['image'].transpose(2, 0, 1).astype(np.float32)
        label = sample['label'].transpose(2, 0, 1)

        return {'image': torch.from_numpy(image.copy()/image.max()), 'label': torch.from_numpy(label.copy()).long()}

