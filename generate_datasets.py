import torch
import torchvision
import random
from PIL import Image
from pathlib import Path

# Python-Fu to import from mnist-c which has a dash in it
import importlib
import os
os.chdir("./mnist-c/")
mnist_c = importlib.import_module("mnist-c.corruptions")
os.chdir("..")

DATA_PATH = "./data"

# Don't change this for reproducible results!!!
random.seed(0)
torch.manual_seed(0)

# Download FashionMNIST train split if not already present
torchvision.datasets.FashionMNIST(root="./data", train=True, download=True)

for i in range(10):
    Path(DATA_PATH + f"/FashionMNIST-Test(C)/{i}").mkdir(parents=True, exist_ok=True)

fashion_train = torchvision.datasets.FashionMNIST(root="./data", train=False, download=True)

for i, (img, label) in enumerate(fashion_train):
    corruption = random.choice(mnist_c.CORRUPTIONS)
    corrupted_data = getattr(mnist_c, corruption)(img)
    corrupted_img = Image.fromarray(corrupted_data).convert("L")
    corrupted_img.save(DATA_PATH + f"/FashionMNIST-Test(C)/{label}/{i}.png")
    if i != 0 and i % 1000 == 0:
        print(f"Completed {i} images")