import torch
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import Dataset

class MNIST_Dataset(Dataset):
    def __init__(self, train=True, transform=None, majority=0.5, sf=False):
        self.transform=transform
        self.train_dataset = datasets.MNIST(
            root="../data",
            train=train,
            download=True,
            transform=self.transform,
        )
        self.classes = self.train_dataset.classes
        self.targets = self.train_dataset.targets


        self.color_map = {
            1: (0., 0., 255.),   # Blue
            0: (255., 0., 0.) # Red
        }

        self.majority_percentage = majority

        # Precompute the indices for the majority and minority splits
        self.index_color_map = self._precompute_indices()
        self.return_sf = sf

    def _precompute_indices(self):
        # Determine the number of samples per class
        targets = self.train_dataset.targets.numpy()
        num_samples = len(targets)
        indices = np.arange(num_samples)

        index_color_map = {}
        np.random.shuffle(indices)
        split_idx = int(len(indices) * self.majority_percentage)
        for idx in indices[split_idx:]:
            index_color_map[idx] = self.color_map[0]

        for idx in indices[:split_idx]:
            index_color_map[idx] = self.color_map[1]

        return index_color_map

    def convert(self, image, id):
        rgb_image = image.clone().repeat(3,1,1)
        c = self.index_color_map[id]
        r,g,b = c[0]/255.0, c[1]/255.0, c[2]/255.0
        image = (image+1)/2
        rgb_image[0] = (image*r*2)-1
        rgb_image[1] = (image*g*2)-1
        rgb_image[2] = (image*b*2)-1

        sf = -1
        if self.return_sf:
            for label, color in self.color_map.items():
                if c == color:
                    sf = label
                    break

        if self.return_sf:
            return rgb_image, sf
        else:
            return rgb_image

    def __len__(self):
        return self.train_dataset.__len__()

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        x,y = self.train_dataset.__getitem__(idx)
        if self.return_sf:
            im, sf = self.convert(x,idx)
            return im, y, sf
        else:
            return self.convert(x,idx),y


def MNIST(data_path, permuted=False, permutation_seed=None):
    channel = 3
    im_size = (28, 28)
    num_classes = 10
    mean = [0.5]
    std = [0.5]
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    if permuted:
        np.random.seed(permutation_seed)
        pixel_permutation = np.random.permutation(28 * 28)
        transform = transforms.Compose(
            [transform, transforms.Lambda(lambda x: x.view(-1, 1)[pixel_permutation].view(1, 28, 28))])

    dst_train = MNIST_Dataset(train=True, transform=transform, majority=0.2) # no augmentation
    dst_test = MNIST_Dataset(train=False, transform=transform, majority=0.2)

    class_names = [str(c) for c in range(num_classes)]
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test


def permutedMNIST(data_path, permutation_seed=None):
    return MNIST(data_path, True, permutation_seed)

