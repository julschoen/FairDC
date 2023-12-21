import torch
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import Dataset

class CelebA_Dataset(Dataset):
    """Face Landmarks dataset."""
    def __init__(self, split='train', transform=None, attributes=['Blond_Hair'], sf=False, s_att=['Male']):
        self.transform = transform
        self.train_dataset = datasets.CelebA(
            root="../data",
            split=split,
            download=False,
            transform=self.transform,
        )



        self.sf = sf

        self.classes = attributes
        self.target_inds = []
        for attr in self.classes:
            self.target_inds.append(self.train_dataset.attr_names.index(attr))

        self.targets = self.train_dataset.attr[:,self.target_inds].squeeze()

        self.s_att = s_att
        self.sens_inds = []
        for attr in self.s_att:
            self.sens_inds.append(self.train_dataset.attr_names.index(attr))

    def __len__(self):
        return self.train_dataset.__len__()

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x,y = self.train_dataset.__getitem__(idx)

        if self.sf:
            return x, y[self.target_inds], y[self.sens_inds]
        else:
            return x, y[self.target_inds]


def CelebA(data_path, permuted=False, permutation_seed=None):
    channel = 3
    im_size = (64, 64)
    num_classes = 2
    mean = [0.5,0.5,0.5]
    std = [0.5,0.5,0.5]
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    

    dst_train = CelebA_Dataset(split='train', transform=transform) # no augmentation
    dst_test = CelebA_Dataset(split='test', transform=transform)

    class_names = [str(c) for c in range(num_classes)]
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test

