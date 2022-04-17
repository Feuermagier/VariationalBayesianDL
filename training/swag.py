from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn.utils.convert_parameters import parameters_to_vector, vector_to_parameters
import torch.nn.functional as F
import math
import numpy as np

@dataclass
class SwagConfig:
    updates_per_epoch: int
    update_every_epochs: int
    max_cov_samples: int

class SWAGWrapper:
    def __init__(self, model: nn.Module, optimizer, config, device, use_lr_cycles: bool = True):
        self.update_every_batches = config.get("update_every_batches", 1)
        self.deviation_samples = config.get("deviation_samples", 10)
        self.start_epoch = config.get("start_epoch", 0)
        self.max_lr = config.get("max_lr", 0.005)
        self.min_lr = config.get("min_lr", 0.001)
        self.model = model
        self.optimizer = optimizer
        self.use_lr_cycles = use_lr_cycles

        self.weights = parameters_to_vector(self.model.parameters())
        self.sq_weights = self.weights**2
        self.updates = 0
        self.deviations = torch.zeros((self.weights.shape[0], self.deviation_samples)).to(device)
        self.param_dist_valid = False
        self.param_dist = None
        self.batches_since_swag_start = 0

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
        if epoch >= self.start_epoch:
            self.batches_since_swag_start += 1

            if self.batches_since_swag_start % self.update_every_batches == 0:
                self.updates += 1
                params = parameters_to_vector(self.model.parameters())
                self.weights = (self.updates * self.weights + params) / (self.updates + 1)
                self.sq_weights = (self.updates * self.sq_weights + params**2) / (self.updates + 1)
                self.deviations = torch.roll(self.deviations, -1, 1)
                self.deviations[:,-1] = params - self.weights
                self.param_dist_valid = False
                if self.use_lr_cycles:
                    print(f"SWAG: Collected a sample at epoch {epoch}, batch {batch_idx} with last lr {self.lr}")

            if self.use_lr_cycles:
                t = 1 - (self.batches_since_swag_start % self.update_every_batches) / self.update_every_batches
                self.lr = t * (self.max_lr - self.min_lr) + self.min_lr
                for g in self.optimizer.param_groups:
                    g["lr"] = self.lr

    def sample(self, input: torch.Tensor):
        old_params = parameters_to_vector(self.model.parameters())
        self.update_param_dist()
        weight_sample = self.param_dist.sample()
        vector_to_parameters(weight_sample, self.model.parameters())
        output = self.model(input)
        vector_to_parameters(old_params, self.model.parameters())
        return output

    def report_status(self):
        print(f"SWAG: Collected {np.minimum(self.updates, self.deviation_samples)} out of {self.deviation_samples} deviation samples and {self.updates} parameter samples")