import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler


def get_train_loader(batch_size):
    transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5), 
        (0.5, 0.5, 0.5))
    ])
    
    train_dataset = datasets.CelebA(
    root="../data",
    split='train',
    download=False,
    transform=transform,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=4, pin_memory=True
    )
    return train_loader

def get_val_loader(batch_size):
    transform=transforms.Compose([
    transforms.ToTensor(),
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
