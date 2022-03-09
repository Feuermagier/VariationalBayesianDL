import numpy as np
import torch
import torchvision
from torchvision.transforms import transforms
import matplotlib.pyplot as plt

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

def trainloader(batch_size: int = 5) -> torch.utils.data.DataLoader:
    dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

def testloader(batch_size: int = 5) -> torch.utils.data.DataLoader:
    dataset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

def flattened_trainloader(batch_size: int = 5) -> torch.utils.data.DataLoader:
    dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=flattening_transform)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

def flattened_testloader(batch_size: int = 5) -> torch.utils.data.DataLoader:
    dataset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=flattening_transform)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

def imshow(img):
    img = img / 2 + 0.5     # denormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))