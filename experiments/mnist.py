import numpy as np
import torch
import torchvision
from torchvision.transforms import transforms
import matplotlib.pyplot as plt

IMAGE_SIZE = 28

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
])

def _flatten(x):
    return torch.flatten(x)

flattening_transform = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize(0,5, 0.5),
  transforms.Lambda(_flatten)
])

def trainloader(batch_size: int = 5, shuffle: bool = True) -> torch.utils.data.DataLoader:
    dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def testloader(batch_size: int = 5, shuffle: bool = True) -> torch.utils.data.DataLoader:
    dataset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)

def fashion_trainloader(batch_size: int = 5, shuffle: bool = True, exclude_classes = []) -> torch.utils.data.DataLoader:
    dataset = torchvision.datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
    indices = torch.full_like(dataset.targets, False)
    for i in range(10):
        if not i in exclude_classes:
            indices |= dataset.targets == i
    dataset.targets = dataset.targets[indices]
    dataset.data = dataset.data[indices]
    print(dataset.data.shape)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def fashion_testloader(batch_size: int = 5, shuffle: bool = True) -> torch.utils.data.DataLoader:
    dataset = torchvision.datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)

def imshow(img):
    #img = img.reshape((IMAGE_SIZE, IMAGE_SIZE, 1)) # unflatten
    img = img / 2 + 0.5     # denormalize
    npimg = img.numpy()
    plt.imshow(npimg, cmap=plt.get_cmap('gray'))