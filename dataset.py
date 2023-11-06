import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np


class MNIST(Dataset):
    def __init__(self, train=True, transform=None, majority=0.5):
        self.train_dataset = datasets.MNIST(
            root="../data",
            train=train,
            download=True,
            transform=transform,
        )

        self.color_map = {
            0: (255., 0., 0.),   # Red
            1: (0, 255, 0),   # Green
            2: (0, 0, 255),   # Blue
            3: (255, 255, 0), # Yellow
            4: (255, 0, 255), # Magenta
            5: (0, 255, 255), # Cyan
            6: (192, 192, 192), # Silver
            7: (194, 125, 25), # Orange
            8: (190., 217., 190.), # Bright Olive
            9: (255., 255., 255.), # White
        }

        self.majority_percentage = majority

        # Precompute the indices for the majority and minority splits
        self.index_color_map = self._precompute_indices()

    def _precompute_indices(self):
        # Determine the number of samples per class
        targets = self.train_dataset.targets.numpy()
        num_samples = len(targets)
        indices = np.arange(num_samples)

        index_color_map = {}
        for class_label, class_color in self.color_map.items():
            print(class_label, class_color)
            class_indices = indices[targets == class_label]
            np.random.shuffle(class_indices)

            # Compute the split index for majority/minority
            split_idx = int(len(class_indices) * self.majority_percentage)

            # Assign the majority color
            for idx in class_indices[:split_idx]:
                index_color_map[idx] = class_color

            # Assign the minority colors uniformly from other classes
            other_colors = [color for label, color in self.color_map.items() if label != class_label]
            for idx in class_indices[split_idx:]:
                index_color_map[idx] = other_colors[np.random.randint(len(other_colors))]

        return index_color_map

    def convert(self, image, id):
        rgb_image = image.clone().repeat(3,1,1)
        r,g,b = self.index_color_map[id]
        r,g,b = r/255.0, g/255.0, b/255.0
        image = (image+1)/2
        rgb_image[0] = (image*r*2)-1
        rgb_image[1] = (image*g*2)-1
        rgb_image[2] = (image*b*2)-1
        return rgb_image

    def __len__(self):
        return self.train_dataset.__len__()

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        x,y = self.train_dataset.__getitem__(idx)
        return self.convert(x,idx),y


def get_train_loader(batch_size):
    transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
            (0.5), 
            (0.5))
    ])
    
    train_dataset = MNIST(transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=4, pin_memory=True
    )
    return train_loader


def get_test_loader(batch_size, shuffle=True):
    transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
            (0.5), 
            (0.5))
    ])
    
    test_dataset = CelebA(train=False, transform=transform)

    return torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, num_workers=4, pin_memory=True
    )
