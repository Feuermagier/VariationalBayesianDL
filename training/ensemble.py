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
        self.models = models

    def state_dict(self):
        return {
            "models": [model.state_dict() for model in self.models]
        }

    def load_state_dict(self, dict):
        for model, state in zip(self.models, dict["models"]):
            model.load_state_dict(state)
        print(self.models)

    def train(self, *args, **kwargs):
        for i, model in enumerate(self.models):
            print(f"Training ensemble member {i}")
            model.train(*args, **kwargs)

    def infer(self, input, *args, **kwargs):
        outputs = []
        for model in self.models:
            outputs.extend(model.infer(input, *args, **kwargs))
        return outputs

    def all_losses(self):
        return [model.all_losses()[0] for model in self.models]