import numpy as np
import torch
import torch.nn as nn
import itertools
from .util import map_activation

class DenseNetwork(nn.Module):
    def __init__(self, units: list, hidden_activation, output_activation):
        super().__init__()
        if len(units) < 2:
            raise ValueError("At least two layers are required (for input and output)")
        
        layers = [nn.Linear(units[i], units[i + 1]) for i in range(len(units) - 1)]
        if len(units) > 2:
            for i in range(len(units) - 3):
                layers.append(nn.Linear(units[i], units[i + 1]))
                print(units[i], units[i + 1])
                layers.append(hidden_activation)
        layers.append(nn.Linear(units[-2], units[-1]))
        layers.append(output_activation)
        self.layers = nn.Sequential(*layers)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.layers(input)