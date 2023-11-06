import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset


class MNIST(Dataset):
    """Face Landmarks dataset."""
    def __init__(self, train=True, transform=None, attributes=['Blond_Hair']):
        self.train_dataset = datasets.MNIST(
            root="../data",
            train=train,
            download=True,
            transform=transform,
        )

        self.classes = attributes
        self.target_inds = []
        for attr in self.classes:
            self.target_inds.append(self.train_dataset.attr_names.index(attr))

    def __len__(self):
        return self.train_dataset.__len__()

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x,y = self.train_dataset.__getitem__(idx)
        print(y.shape)
        return x,y


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
