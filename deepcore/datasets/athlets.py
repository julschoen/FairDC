import torch
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import Dataset
import pandas as pd


class Athlets_Dataset(Dataset):
    def __init__(self, train=True, transform=None, sf=False, s_att=['Gender']):
        self.data_path = '../data/basket_volley'
        self.meta = pd.read_csv(os.path.join(self.data_path,'splits.csv')).drop(columns='Unnamed: 0')
        self.meta = self.meta[self.meta['Train']==train].reset_index()
        self.sf = sf

        self.classes = ['basket', 'volley']

        self.transform = transforms.Resize((64,64), antialias=True)

        self.class_num = dict()
        i = 0
        for c in self.classes:
            self.class_num[c] = i
            i += 1

        self.s_att = s_att
      

    def __len__(self):
        return self.meta.shape[0]

    def __getitem__(self, idx):
        im = io.imread(self.meta.loc[idx]['Path'])
        im = im.transpose((2, 0, 1))/255.
        im = (im*2)-1
        im = torch.from_numpy(im).float()
        im = self.transform(im)
        

            
        target = self.meta.loc[idx]['Target']
        if im.shape != (3,64,64):
            im = im[:3]
        if self.sf:
            return im, target, self.meta.loc[idx][self.s_att]
        else:
            return im, target


def Athlets(data_path, permuted=False, permutation_seed=None):
    channel = 3
    im_size = (64, 64)
    num_classes = 2
    mean = [0.5,0.5,0.5]
    std = [0.5,0.5,0.5]

    dst_train = Athlets_Dataset() # no augmentation
    dst_test = Athlets_Dataset(train=False)

    class_names = [str(c) for c in range(num_classes)]
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test

