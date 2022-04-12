from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

@dataclass
class SwagConfig:
    updates_per_epoch: int
    update_every_epochs: int
    max_cov_samples: int

class SWAGWrapper:
    def __init__(self, model: nn.Module, config, device):
        self.update_every_batches = config.get("update_every_batches", 1)
        self.update_every_epochs = config.get("update_every_epochs", 1)
        self.deviation_samples = config.get("deviation_samples", 10)
        self.start_epoch = config.get("start_epoch", 0)
        self.model = model

        self.weights = torch.nn.utils.convert_parameters.parameters_to_vector(self.model.parameters())
        self.sq_weights = self.weights**2
        self.updates = 0
        self.deviations = torch.zeros((self.weights.shape[0], self.deviation_samples)).to(device)
        self.param_dist_valid = False
        self.param_dist = None

    @property
    def mean(self):
        return self.weights
    
    def update_param_dist(self):
        if not self.param_dist_valid:
            diag = 0.5 * (torch.relu(self.sq_weights - self.weights**2) + 1e-6) # Adding 1e-6 for numerical stability
            cov_factor = self.deviations / math.sqrt(2 * (self.deviation_samples - 1))
            #self.param_dist = torch.distributions.MultivariateNormal(self.mean, cov_factor, diag)
            self.param_dist = torch.distributions.LowRankMultivariateNormal(self.mean, cov_factor, diag)
            self.param_dist_valid = True

    def update(self, epoch, batch_idx):
        if epoch == self.start_epoch and (batch_idx + 1) == self.update_every_batches:
            print(f"SWAG: starting to collect samples at epoch {epoch}, batch {batch_idx}")
        if epoch >= self.start_epoch and (epoch + 1) % self.update_every_epochs == 0 and (batch_idx + 1) % self.update_every_batches == 0:
            self.updates += 1
            params = torch.nn.utils.convert_parameters.parameters_to_vector(self.model.parameters())
            self.weights = (self.updates * self.weights + params) / (self.updates + 1)
            self.sq_weights = (self.updates * self.sq_weights + params**2) / (self.updates + 1)
            self.deviations = torch.roll(self.deviations, -1, 1)
            self.deviations[:,-1] = params - self.weights
            self.param_dist_valid = False

    def sample(self, model: nn.Module, input: torch.Tensor):
        self.update_param_dist()
        weight_sample = self.param_dist.sample()
        torch.nn.utils.convert_parameters.vector_to_parameters(weight_sample, model.parameters())
        return model(input)