from types import LambdaType
from typing import Callable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def combined_variance_output(input: torch.Tensor, ensemble: list[nn.Module]) -> tuple[torch.Tensor, torch.Tensor]:
    means = torch.zeros((input.shape[0], len(ensemble)))
    variances = torch.zeros((input.shape[0], len(ensemble)))
    for i, model in enumerate(ensemble):
        output = model(input)
        means[:,i] = output[:,0]
        variances[:,i] = torch.log1p(torch.exp(output[:,1])) + 10e-6
    mean = torch.mean(means, dim=1)
    # Variance from https://github.com/cameronccohen/deep-ensembles/blob/master/Tutorial.ipynb
    variance = torch.mean(variances + means**2, dim=1) - mean**2 
    return mean, variance


class SimpleEnsemble(nn.Module):
    def __init__(self, module_init: Callable[[], nn.Module], count: int):
        self.modules = [module_init() for _ in range(count)]
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = torch.tensor([module(input) for module in self.modules])
        return output.mean()