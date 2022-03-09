import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import math
import time
from .util import GaussianMixture

def run_bbb_epoch(model: nn.Sequential, optimizer: torch.optim.Optimizer, loss_fn, loader: torch.utils.data.DataLoader, device: torch.device) -> torch.Tensor:
    model.train()
    epoch_loss = torch.tensor(0, dtype=torch.float)
    for i, (data, target) in enumerate(loader):
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = model(data)
        pi = (2**(len(loader) - i)) / (2**len(loader) - 1)
        kl = sum([getattr(layer, "kl", 0) for layer in model])
        loss = pi * kl + loss_fn(output, target)
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()
        epoch_loss += loss.cpu()
    return epoch_loss

class BBBLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, weight_prior, bias_prior: GaussianMixture, device: torch.device, **kwargs):
        super().__init__()
        self.sampling = kwargs.get("sampling", "local_reparametrization")
        self.deterministic_eval = kwargs.get("deterministic_eval", False)
        self.device = device

        self.in_features, self.out_features = in_features, out_features
        self.weight_prior, self.bias_prior = weight_prior, bias_prior
        self.log_prior, self.log_posterior = 0, 0

        # Weights
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.1, 0.1))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-3, -2))

        # Biases
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.1, 0.1))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-3, -2))

    def sample_parameters(self, mu, rho):
        epsilon = torch.empty(mu.shape).normal_(0, 1).to(self.device)
        return mu + to_sigma(rho) * epsilon

    def forward(self, input: torch.Tensor):
        if not self.training and self.deterministic_eval:
            weight = self.weight_mu
            bias = self.bias_mu
            self.kl = 0
            return F.linear(input, weight, bias)
        elif self.sampling == "parameters":
            weight = self.sample_parameters(self.weight_mu, self.weight_rho)
            bias = self.sample_parameters(self.bias_mu, self.bias_rho)

            log_prior = self.weight_prior.log_prob(weight).sum() + self.bias_prior.log_prob(bias).sum()
            log_posterior = log_prob(self.weight_mu, self.weight_rho, weight).sum() + log_prob(self.bias_mu, self.bias_rho, bias).sum()
            self.kl = log_posterior - log_prior

            return F.linear(input, weight, bias)
        elif self.sampling == "local_reparametrization":
            activation_mu = F.linear(input, self.weight_mu, self.bias_mu)
            activation_var = F.linear(input**2, to_sigma(self.weight_rho)**2, to_sigma(self.bias_rho)**2)
            activation_std = torch.sqrt(activation_var)
            
            # The following line is extremely expensive
            log_prior = self.weight_prior.log_prob(self.weight_mu).sum() + self.bias_prior.log_prob(self.bias_mu).sum() 
            self.kl = -log_prior

            epsilon = torch.empty(activation_mu.shape).normal_(0, 1).to(self.device)
            return activation_mu + activation_std * epsilon
        else:
            raise ValueError("Invalid value of weight_draw")

def to_sigma(rho):
    return torch.log1p(torch.exp(rho)) + 1e-6

def log_prob(mu, rho, value):
    sigma = to_sigma(rho)
    return -((value - mu)**2) / (2 * sigma**2) - sigma.log() - math.log(math.sqrt(2 * math.pi))