from types import LambdaType
from typing import Callable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def combined_variance_output(input, ensemble):
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

class Ensemble(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return {
            "models": [model.state_dict(destination, prefix, keep_vars) for model in self.models]
        }

    def load_state_dict(self, dict):
        for model, state in zip(self.models, dict["models"]):
            model.load_state_dict(state)

    def train_model(self, *args, **kwargs):
        for i, model in enumerate(self.models):
            print(f"Training ensemble member {i}")
            model.train_model(*args, **kwargs)

    def infer(self, input, samples, *args, **kwargs):
        assert samples >= len(self.models)
        outputs = []
        for i, model in enumerate(self.models):
            if i == len(self.models) - 1:
                outputs.append(model.infer(input, samples - i * (samples // len(self.models)), *args, **kwargs))
            else:
                outputs.append(model.infer(input, samples // len(self.models), *args, **kwargs))
        return torch.concat(outputs)

    def all_losses(self):
        return [model.all_losses()[0] for model in self.models]