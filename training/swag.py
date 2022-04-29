from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn.utils.convert_parameters import parameters_to_vector, vector_to_parameters
import torch.nn.functional as F
import math
import numpy as np
from .util import generate_model

class SwagModel(nn.Module):
    def __init__(self, layers, config):
        super().__init__()
        self.model = generate_model(layers)
        self.losses = []
        self.update_every_batches = config.get("update_every_batches", 1)
        self.deviation_samples = config.get("deviation_samples", 10)
        self.start_epoch = config.get("start_epoch", 0)
        self.use_lr_cycles = config.get("use_lr_cycles", False)
        self.max_lr = config.get("max_lr", 0.005)
        self.min_lr = config.get("min_lr", 0.001)

        self.weights = parameters_to_vector(self.model.parameters())
        self.sq_weights = self.weights**2
        self.updates = 0
        self.deviations = torch.zeros((self.weights.shape[0], self.deviation_samples))
        self.param_dist_valid = False
        self.param_dist = None
        self.batches_since_swag_start = 0

    def state_dict(self):
        return {
            "model": self.model.state_dict(),
            "losses": self.losses,
            "updates": self.updates,
            "batches_since_swag_start": self.batches_since_swag_start,
            "weights": self.weights,
            "sq_weights": self.sq_weights,
            "deviations": self.deviations
        }

    def load_state_dict(self, state):
        self.model.load_state_dict(state["model"])
        self.losses = state["losses"]
        self.updates = state["updates"]
        self.batches_since_swag_start = state["batches_since_swag_start"]
        self.weights = state["weights"]
        self.sq_weights = state["sq_weights"]
        self.deviations = state["deviations"]
        self.param_dist_valid = False

    def train(self, epochs, loss_fn, optimizer_factory, loader, batch_size, device, report_every_epochs=1):
        self.model.to(device)
        self.model.train()
        optimizer = optimizer_factory(self.model.parameters())

        for epoch in range(epochs):
            epoch_loss = torch.tensor(0, dtype=torch.float)
            for batch_idx, (data, target) in enumerate(loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = loss_fn(output, target)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.cpu()
                self.swag_update(epoch, batch_idx)
            epoch_loss /= (len(loader) * batch_size)
            self.losses.append(epoch_loss.detach())
            if report_every_epochs > 0 and epoch % report_every_epochs == 0:
                print(f"Epoch {epoch}: loss {epoch_loss}")
                self.report_status()
        if report_every_epochs >= 0:
            print(f"Final loss {epoch_loss}")

    def infer(self, input, samples):
        if samples <= 0:
            return [self.model(input)]
        
        old_params = parameters_to_vector(self.model.parameters())
        self.update_param_dist()
        outputs = []
        for _ in range(samples):
            weight_sample = self.param_dist.sample().to(input.device)
            vector_to_parameters(weight_sample, self.model.parameters())
            outputs.append(self.model(input))
        vector_to_parameters(old_params, self.model.parameters())
        return torch.stack(outputs)

    def all_losses(self):
        return [self.losses]

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

    def swag_update(self, epoch, batch_idx):
        if epoch >= self.start_epoch:
            self.batches_since_swag_start += 1

            if self.batches_since_swag_start % self.update_every_batches == 0:
                self.updates += 1
                params = parameters_to_vector(self.model.parameters()).cpu()
                self.weights = (self.updates * self.weights + params) / (self.updates + 1)
                self.sq_weights = (self.updates * self.sq_weights + params**2) / (self.updates + 1)
                self.deviations = torch.roll(self.deviations, -1, 1)
                self.deviations[:,-1] = params - self.weights
                self.param_dist_valid = False
                if self.use_lr_cycles:
                    print(f"SWAG: Collected a sample at epoch {epoch}, batch {batch_idx}")

            if self.use_lr_cycles:
                t = 1 - (self.batches_since_swag_start % self.update_every_batches) / self.update_every_batches
                self.lr = t * (self.max_lr - self.min_lr) + self.min_lr
                for g in self.optimizer.param_groups:
                    g["lr"] = self.lr

    def report_status(self):
        print(f"SWAG: Collected {np.minimum(self.updates, self.deviation_samples)} out of {self.deviation_samples} deviation samples and {self.updates} parameter samples")

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
                    print(f"SWAG: Collected a sample at epoch {epoch}, batch {batch_idx}")

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
