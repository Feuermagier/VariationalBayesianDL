import torch
import torchvision
import random
from PIL import Image
from pathlib import Path
import urllib.request
import zipfile
import sys

# Python-Fu to import from mnist-c which has a dash in it
import importlib
import os
os.chdir("./mnist-c/")
mnist_c = importlib.import_module("mnist-c.corruptions")
os.chdir("..")

DATA_PATH = "./data/" if len(sys.argv) <= 1 else sys.argv[1]
print(f"Creating datasets in {DATA_PATH}")

# Don't change this for reproducible results!!!
random.seed(0)
torch.manual_seed(0)

# Download CIFAR10
torchvision.datasets.CIFAR10(root=DATA_PATH, train=False, download=True)
torchvision.datasets.CIFAR10(root=DATA_PATH, train=True, download=True)

# Download FashionMNIST test and train splits if not already present
torchvision.datasets.FashionMNIST(root=DATA_PATH, train=False, download=True)
torchvision.datasets.FashionMNIST(root=DATA_PATH, train=True, download=True)

# Generate FashionMNIST(C) if not present
if not Path(DATA_PATH + "FashionMNIST-Test(C)").exists():
    for i in range(10):
        Path(DATA_PATH + f"FashionMNIST-Test(C)/{i}").mkdir(parents=True, exist_ok=True)

    fashion_train = torchvision.datasets.FashionMNIST(root=DATA_PATH, train=False, download=True)

    for i, (img, label) in enumerate(fashion_train):
        corruption = random.choice(mnist_c.CORRUPTIONS)
        corrupted_data = getattr(mnist_c, corruption)(img)
        corrupted_img = Image.fromarray(corrupted_data).convert("L")
        corrupted_img.save(DATA_PATH + f"FashionMNIST-Test(C)/{label}/{i}.png")
        if i != 0 and i % 1000 == 0:
            print(f"Completed {i} images")


# Download UCI datasets
Path(DATA_PATH + "UCI/").mkdir(parents=True, exist_ok=True)

datasets = {
    "housing": "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data",
    "concrete": "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls",
    "energy": "http://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx",
    "power": "https://archive.ics.uci.edu/ml/machine-learning-databases/00294/CCPP.zip",
    "wine": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
    "yacht": "http://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data"
}

for name, url in datasets.items():
    filename = url.split('/')[-1]
    if not Path(DATA_PATH + "UCI/" + filename).exists():
        print(f"Downloading {name}...")
        urllib.request.urlretrieve(url, DATA_PATH + "UCI/" + filename)

print(f"Extracting dataset...")
zipfile.ZipFile(DATA_PATH + "UCI/CCPP.zip").extractall(DATA_PATH + "UCI/")