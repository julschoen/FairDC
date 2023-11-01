import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset


class CelebA(Dataset):
    """Face Landmarks dataset."""
    def __init__(self, split='train', transform=None, attributes=['Blond_Hair']):
        self.train_dataset = datasets.CelebA(
            root="../data",
            split=split,
            download=False,
            transform=transform,
        )

        self.target_inds = []
        for attr in attributes:
            self.target_inds.append(self.train_dataset.attr_names.index(attr))

    def __len__(self):
        return self.train_dataset.__len__()

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x,y = self.train_dataset.__getitem__(idx)
        
        return x,y[:,self.target_inds]


def get_train_loader(batch_size):
    transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((64,64), antialias=True),
    transforms.Normalize(
        (0.5, 0.5, 0.5), 
        (0.5, 0.5, 0.5))
    ])
    
    train_dataset = CelebA(transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=4, pin_memory=True
    )
    return train_loader

def get_val_loader(batch_size):
    transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((64,64), antialias=True),
    transforms.Normalize(
        (0.5, 0.5, 0.5), 
        (0.5, 0.5, 0.5))
    ])
    
    valid_dataset = datasets.CelebA(
    root="../data",
    split='valid',
    download=False,
    transform=transform,
    )

    return torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, num_workers=4, pin_memory=True
    )

def get_test_loader(batch_size, shuffle=True):
    transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((64,64), antialias=True),
    transforms.Normalize(
        (0.5, 0.5, 0.5), 
        (0.5, 0.5, 0.5))
    ])
    
    test_dataset = datasets.CelebA(
    root="../data",
    split='test',
    download=False,
    transform=transform,
    )

    return torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, num_workers=4, pin_memory=True
    )
