import numpy as np
import torch
import torchvision
from torchvision.transforms import transforms

# From https://github.com/izmailovpavel/understandingbdl/blob/5d1004896ea4eb674cff1c2088dc49017a667e9e/swag/models/preresnet.py
transform_train = transforms.Compose([
    transforms.Resize(32),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


def cifar10_trainloader(path, batch_size: int = 4, shuffle: bool = True, exclude_classes = []):
    dataset = torchvision.datasets.CIFAR10(root=path, train=True, download=True, transform=transform_train)
    dataset.targets = torch.tensor(dataset.targets)
    _select_classes(dataset, exclude_classes)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)

def cifar10_testloader(path, batch_size: int = 4, shuffle: bool = True, exclude_classes = []):
    dataset = torchvision.datasets.CIFAR10(root=path, train=False, download=True, transform=transform_test)
    dataset.targets = torch.tensor(dataset.targets)
    _select_classes(dataset, exclude_classes)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)

def _select_classes(dataset, exclude_classes):
    indices = torch.full_like(dataset.targets, False, dtype=torch.bool)
    for i in range(10):
        if not i in exclude_classes:
            indices |= dataset.targets == i
    dataset.targets = dataset.targets[indices]
    dataset.data = dataset.data[indices]