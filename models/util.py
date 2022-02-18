import torch
import torch.nn as nn

def map_activation(name: str):
    if name == "relu":
        return nn.ReLU()
    elif name == "selu":
        return nn.SELU()
    elif name == "softmax":
        return nn.Softmax()
    else:
        raise ValueError(f"Unknown activation function '{name}'")