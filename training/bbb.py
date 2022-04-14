from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import math
import time
from .util import GaussianMixture

def run_bbb_epoch(model: nn.Sequential, optimizer: torch.optim.Optimizer, loss_fn, loader: torch.utils.data.DataLoader, device: torch.device, **kwargs) -> torch.Tensor:
    samples = kwargs.get("samples", 1)
    kl_rescaling = kwargs.get("kl_rescaling", 1)
    model.train()
    epoch_loss = torch.tensor(0, dtype=torch.float)
    for i, (data, target) in enumerate(loader):
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        loss = torch.tensor(0, dtype=torch.float)
        pi = kl_rescaling / len(loader)
        #pi = (2**(len(loader) - i - 1)) / (2**len(loader) - 1)

        for _ in range(samples):
            output = model(data)
            kl = sum([getattr(layer, "kl", 0) for layer in model])
            loss += (pi * kl + loss_fn(output, target)).cpu()
        loss /= samples
        loss.backward()
        nn.utils.clip_grad.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()
        epoch_loss += loss.cpu()
    return epoch_loss


class BBBLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, weight_prior, bias_prior, device: torch.device, **kwargs):
        super().__init__()
        self.is_bayesian = True
        self.sampling = kwargs.get("sampling", "activations")
        self.mc_sample = kwargs.get("mc_sample", 1)
        self.freeze_on_eval = kwargs.get("freeze_on_eval", True)
        self.device = device
        self.in_features, self.out_features = in_features, out_features
        self.weight_prior, self.bias_prior = weight_prior, bias_prior

        # Weights
        self.weight_mu = nn.Parameter(torch.empty((self.out_features, self.in_features)).normal_(0, 0.1))
        self.weight_rho = nn.Parameter(torch.empty((self.out_features, self.in_features)).uniform_(-3, -3))

        # Biases
        self.bias_mu = nn.Parameter(torch.empty(out_features).normal_(0, 0.1))
        self.bias_rho = nn.Parameter(torch.empty(self.out_features).uniform_(-3, -3))

        self.kl = 0

    def sample_parameters(self, mu, rho):
        epsilon = torch.empty(rho.shape).normal_(0, 1).to(self.device)
        return mu + to_sigma(rho) * epsilon

    def forward(self, input: torch.Tensor):
        self.kl = 0

        if self.sampling == "parameters":
            output = torch.zeros((input.shape[0], self.out_features))

            for i in range(self.mc_sample):
                weight = self.sample_parameters(self.weight_mu, self.weight_rho)
                bias = self.sample_parameters(self.bias_mu, self.bias_rho)

                log_prior = self.weight_prior.log_prob(weight).sum() + self.bias_prior.log_prob(bias).sum()
                log_posterior = log_prob(self.weight_mu, self.weight_rho, weight).sum() + log_prob(self.bias_mu, self.bias_rho, bias).sum()
                self.kl += log_posterior - log_prior
                
                output += F.linear(input, weight, bias)

            self.kl /= self.mc_sample
            return output / self.mc_sample
        elif self.sampling == "activations":
            activation_mu = F.linear(input, self.weight_mu, self.bias_mu)
            activation_var = F.linear(input**2, to_sigma(self.weight_rho)**2, to_sigma(self.bias_rho)**2)
            activation_std = torch.sqrt(activation_var)
            
            #log_prior = self.weight_prior.log_prob(self.weight_mu).sum() + self.bias_prior.log_prob(self.bias_mu).sum() 
            #self.kl = -log_prior
            weight_kl = self.weight_prior.kl_divergence(self.weight_mu, to_sigma(self.weight_rho))
            bias_kl = self.bias_prior.kl_divergence(self.bias_mu, to_sigma(self.bias_rho))
            self.kl = weight_kl + bias_kl

            if not self.training and self.freeze_on_eval:
                epsilon = torch.empty(activation_mu.shape[1:]).normal_(0, 1).unsqueeze(0).expand((activation_mu.shape)).to(self.device)
            else:
                epsilon = torch.empty(activation_mu.shape).normal_(0, 1).to(self.device)
            output = activation_mu + activation_std * epsilon
                # How to calculate the log posterior?
            
            return output / self.mc_sample
        else:
            raise ValueError("Invalid value of sampling")

class BBBConvolution(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, weight_prior, bias_prior, device: torch.device, **kwargs):
        super().__init__()
        self.is_bayesian = True
        self.sampling = kwargs.get("sampling", "activations")
        self.stride = kwargs.get("stride", 1)
        self.freeze_on_eval = kwargs.get("freeze_on_eval", True)
        self.device = device
        self.out_channels, self.in_channels = out_channels, in_channels
        self.kernel_size = kernel_size

        self.weight_prior, self.bias_prior = weight_prior, bias_prior

        # Weights
        self.weight_mu = nn.Parameter(torch.empty((self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)).normal_(0, 0.1))
        self.weight_rho = nn.Parameter(torch.empty((self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)).uniform_(-3, -3))

        # Biases
        self.bias_mu = nn.Parameter(torch.empty(self.out_channels).normal_(0, 0.1))
        self.bias_rho = nn.Parameter(torch.empty(self.out_channels).uniform_(-3, -3))

        self.kl = 0

    def sample_parameters(self, mu, rho):
        epsilon = torch.empty(rho.shape).normal_(0, 1).to(self.device)
        return mu + to_sigma(rho) * epsilon

    def forward(self, input: torch.Tensor):
        self.kl = 0

        if self.sampling == "parameters":
            raise NotImplementedError()
        elif self.sampling == "activations":
            activation_mu = F.conv2d(input, self.weight_mu, self.bias_mu, self.stride)
            activation_var = F.conv2d(input**2, to_sigma(self.weight_rho)**2, to_sigma(self.bias_rho)**2, self.stride)
            activation_std = torch.sqrt(activation_var)
            
            #log_prior = self.weight_prior.log_prob(self.weight_mu).sum() + self.bias_prior.log_prob(self.bias_mu).sum() 
            #self.kl = -log_prior
            weight_kl = self.weight_prior.kl_divergence(self.weight_mu, to_sigma(self.weight_rho))
            bias_kl = self.bias_prior.kl_divergence(self.bias_mu, to_sigma(self.bias_rho))
            self.kl = weight_kl + bias_kl

            if not self.training and self.freeze_on_eval:
                epsilon = torch.empty(activation_mu.shape[1:]).normal_(0, 1).unsqueeze(0).expand((activation_mu.shape)).to(self.device)
            else:
                epsilon = torch.empty(activation_mu.shape).normal_(0, 1).to(self.device)
            output = activation_mu + activation_std * epsilon
            
            return output
        else:
            raise ValueError("Invalid value of sampling")

def to_sigma(rho):
    return F.softplus(rho)

def log_prob(mu, rho, value):
    #sigma = to_sigma(rho)
    #return torch.clamp(-((value - mu)**2) / (2 * sigma**2) - sigma.log() - math.log(math.sqrt(2 * math.pi)), -23, 0)
    return torch.clamp(torch.distributions.Normal(mu, to_sigma(rho), False).log_prob(value), -23, 0)

# Closed form KL divergence for gaussians
# See https://github.com/kumar-shridhar/PyTorch-BayesianCNN/blob/master/metrics.py
def gauss_kl(mu_q, sig_q, mu_p, sig_p):
    kl = 0.5 * (2 * torch.log(sig_p / sig_q) - 1 + (sig_q / sig_p).pow(2) + ((mu_p - mu_q) / sig_p).pow(2)).sum()
    return kl

class GaussianPrior:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        self.dist = torch.distributions.Normal(mu, sigma)

    def log_prob(self, x):
        return self.dist.log_prob(x)

    def kl_divergence(self, mu2, sigma2):
        #kl = 0.5 * (2 * torch.log(sigma2 / self.sigma) - 1 + (self.sigma / sigma2).pow(2) + ((mu2 - self.mu) / sigma2).pow(2))
        kl = 0.5 * (2 * torch.log(self.sigma / sigma2) - 1 + (sigma2 / self.sigma).pow(2) + ((self.mu - mu2) / self.sigma).pow(2))
        return kl.sum()

class MixturePrior:
    def __init__(self, pi, sigma1, sigma2, validate_args=None):
        self.pi = torch.tensor(pi)
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.dist1 = torch.distributions.Normal(0, sigma1, validate_args)
        self.dist2 = torch.distributions.Normal(0, sigma2, validate_args)

    def log_prob(self, value):
        prob1 = torch.log(self.pi) + torch.clamp(self.dist1.log_prob(value), -23, 0)
        prob2 = torch.log(1 - self.pi) + torch.clamp(self.dist2.log_prob(value), -23, 0)
        return torch.logaddexp(prob1, prob2)

    def kl_divergence(self, mu2, sigma2):
        return -self.log_prob(mu2).sum()
