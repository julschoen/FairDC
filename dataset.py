import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset


class MNIST(Dataset):
    def __init__(self, train=True, transform=None):
        self.train_dataset = datasets.MNIST(
            root="../data",
            train=train,
            download=True,
            transform=transform,
        )

        self.color_map = {
        0: (255, 0, 0),   # Red
        1: (0, 255, 0),   # Green
        2: (0, 0, 255),   # Blue
        3: (255, 255, 0), # Yellow
        4: (255, 0, 255), # Magenta
        5: (0, 255, 255), # Cyan
        6: (192, 192, 192), # Silver
        7: (128, 0, 128), # Purple
        8: (128, 128, 0), # Olive
        9: (0, 128, 128), # Teal
    }

    def convert(self, image, label):
        image = image.repeat(3,1,1)  # Convert grayscale to RGB
        # Apply the color corresponding to the class label
        r, g, b = self.color_map[label]
        colored_image = transforms.functional.adjust_hue(image, r/255.0)
        colored_image = transforms.functional.adjust_saturation(colored_image, g/255.0)
        colored_image = transforms.functional.adjust_brightness(colored_image, b/255.0)
        return colored_image, label

    def __len__(self):
        return self.train_dataset.__len__()

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        x,y = self.train_dataset.__getitem__(idx)
        return self.convert(x,y)


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
