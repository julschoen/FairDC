import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset


class CelebA(Dataset):
    """Face Landmarks dataset."""
    def __init__(self, split='train', transform=None):
        self.train_dataset = datasets.CelebA(
            root="../data",
            split=split,
            download=False,
            transform=transform,
        )

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks], dtype=float).reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample


def get_train_loader(batch_size):
    transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((64,64), antialias=True),
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
    print(train_dataset.attr)

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
