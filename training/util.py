import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def gauss_logprob(mean: torch.Tensor, variance: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    var = torch.max(torch.tensor(1e-6), variance)
    return -((x - mean) ** 2) / (2 * var) - torch.log(var) - math.log(math.sqrt(2 * math.pi))

# Weighted sum of two gaussian distributions
class GaussianMixture:
    def __init__(self, pi, sigma1, sigma2):
        self.pi, self.sigma1, self.sigma2 = pi, sigma1, sigma2
        self.gaussian1 = torch.distributions.Normal(0, sigma1)
        self.gaussian2 = torch.distributions.Normal(0, sigma2)

    def log_prob(self, value):
        p1 = torch.exp(self.gaussian1.log_prob(value)) + 1e-6
        p2 = torch.exp(self.gaussian2.log_prob(value)) + 1e-6
        return torch.log((self.pi * p1 + (1 - self.pi) * p2)).sum()

class GaussWrapper(nn.Module):
    def __init__(self, mean: nn.Module, var: torch.Tensor, learn_var: bool = False):
        super().__init__()
        self.mean = mean
        self.rho = torch.log(torch.exp(var) - 1)
        if learn_var:
            self.rho = nn.Parameter(self.rho)

    def forward(self, input):
        return self.mean(input), F.softplus(self.rho).repeat(input.shape[0])

    def __iter__(self):
        return iter(self.mean)