import numpy as np
import torch
import torchvision
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import random

from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader

def trainloader(path, dataset):
    dataset = get_dataset(dataset=dataset, download=True, root_dir=path)
    return get_train_loader("standard")