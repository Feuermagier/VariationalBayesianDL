import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import itertools

from training.bbb_layers import BBBLinear
from training.dropout import FixableDropout

def map_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "tanh":
        return nn.Tanh()
    elif name == "logsoftmax":
        return nn.LogSoftmax(dim=1)
    else:
        raise ValueError(f"Unknown activation function {name}")

def generate_model(architecture, print_summary=False):
    layers = []
    for i, (ty, size) in enumerate(architecture):
        if ty == "pool":
            layers.append(nn.MaxPool2d(size))
        elif ty == "flatten":
            layers.append(nn.Flatten())
        elif ty == "relu":
            layers.append(nn.ReLU())
        elif ty == "sigmoid":
            layers.append(nn.Sigmoid())
        elif ty == "logsoftmax":
            layers.append(nn.LogSoftmax(dim=1))
        elif ty == "fc":
            (in_features, out_features) = size
            layers.append(nn.Linear(in_features, out_features))
        elif ty == "v_fc":
            (in_features, out_features, prior) = size
            layers.append(BBBLinear(in_features, out_features, prior, prior, initialization="blundell"))
        elif ty == "conv":
            (in_channels, out_channels, kernel_size) = size
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size))
        elif ty == "dropout":
            p, = size
            layers.append(FixableDropout(p))
        else:
            raise ValueError(f"Unknown layer type '{ty}'")
    model = nn.Sequential(*layers)

    if print_summary:
        print(f"Generated model: {model}")
        print(f"{sum([p.numel() for p in model.parameters() if p.requires_grad])} trainable parameters")
        
    return model