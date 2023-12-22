import torch
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import os
from skimage import io

class HAM10000_Dataset(Dataset):
    def __init__(self, train=True, transform=None, sf=False, s_att=['sex']):
        self.data_path = '../data/HAM10000'
        self.meta = pd.read_csv(os.path.join(self.data_path,'HAM10000_metadata.csv'))
        if train:
            id_path = os.path.join(self.data_path, 'train_ids.npy')
        else:
            id_path = os.path.join(self.data_path, 'test_ids.npy')
        self.ids = np.load(id_path)
        self.sf = sf

        self.transform = transforms.Resize((64,64), antialias=True)

        self.classes = ['nv','bkl','mel','bcc','akiec','vasc','df']

        self.class_num = dict()
        i = 0
        for c in self.classes:
            self.class_num[c] = i
            i += 1

        self.s_att = s_att
      

    def __len__(self):
        return self.ids.shape[0]

    def __getitem__(self, idx):
        im = io.imread(os.path.join(self.data_path, 'images', self.meta.loc[self.ids[idx]]['image_id']+'.jpg'))
        im = im.transpose((2, 0, 1))/255.
        im = (im*2)-1
        im = torch.from_numpy(im).float()
        im = self.transform(im)
        target = self.class_num[self.meta.loc[self.ids[idx]]['dx']]
        
        if self.sf:
            return im, target, self.meta.loc[self.ids[idx]][self.s_att]
        else:
            return im, target


def HAM10000(data_path, permuted=False, permutation_seed=None):
    channel = 3
    im_size = (64, 64)
    num_classes = 7
    mean = [0.5,0.5,0.5]
    std = [0.5,0.5,0.5]

    dst_train = HAM10000_Dataset(train=True) # no augmentation
    dst_test = HAM10000_Dataset(train=False)

    class_names = [str(c) for c in range(num_classes)]
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test

