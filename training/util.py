import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import itertools

def gauss_logprob(mean: torch.Tensor, variance: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    #var = torch.max(torch.tensor(1e-6), variance)
    return -((x - mean) ** 2) / (2 * variance) - torch.log(variance.sqrt()) - math.log(math.sqrt(2 * math.pi))

def sgd(lr):
    return lambda parameters: torch.optim.SGD(parameters, lr=lr)

def adam(lr):
    return lambda parameters: torch.optim.Adam(parameters, lr=lr)

def nll_loss(output, target, reduction="mean"):
    mean = output[...,0]
    std = output[...,1]
    return F.gaussian_nll_loss(mean, target, std**2, reduction)

# Weighted sum of two gaussian distributions
class GaussianMixture:
    def __init__(self, pi, sigma1, sigma2):
        self.log_pi, self.sigma1, self.sigma2 = torch.log(torch.tensor(pi)), sigma1, sigma2
        self.gaussian1 = torch.distributions.Normal(0, sigma1)
        self.gaussian2 = torch.distributions.Normal(0, sigma2)

    def log_prob(self, value):
        #p1 = torch.exp(self.gaussian1.log_prob(value)) + 1e-6
        #p2 = torch.exp(self.gaussian2.log_prob(value)) + 1e-6
        #return torch.log((self.pi * p1 + (1 - self.pi) * p2)).sum()
        return torch.logaddexp(self.log_pi + self.gaussian1.log_prob(value), self.log_pi + self.gaussian2.log_prob(value))

# class GaussWrapper(nn.Module):
#     def __init__(self, mean, std_init: torch.Tensor, learn_var: bool = False):
#         super().__init__()
#         self.mean = mean
#         self.rho = torch.log(torch.exp(std_init) - 1)
#         if learn_var:
#             self.rho = nn.Parameter(self.rho)
#         self.learn_var = learn_var

#     def forward(self, input):
#         print(F.softplus(self.rho))
#         return self.mean(input), F.softplus(self.rho).repeat(input.shape[0])

#     def state_dict(self, destination=None, prefix='', keep_vars=False):
#         return {
#             "mean": self.mean.state_dict(destination, prefix, keep_vars),
#             "rho": self.rho
#         }

#     def load_state_dict(self, state):
#         self.mean.load_state_dict(state["mean"])
#         self.rho = state["rho"]

#     def train_model(self, epochs, optimizer_factory, loss_reduction, *args, **kwargs):
#         if self.learn_var:
#             optimizer_factory_ext = lambda p: optimizer_factory(list(p) + [self.rho])
#         else:
#             optimizer_factory_ext = optimizer_factory
        
#         loss_fn = lambda output, target: F.gaussian_nll_loss(output, target, F.softplus(self.rho).repeat(output.shape[0])**2, reduction=loss_reduction)
#         return self.mean.train_model(epochs, loss_fn, optimizer_factory_ext, *args, **kwargs)
        
#     def infer(self, input, samples):
#         means = self.mean.infer(input, samples)
#         return torch.stack((means, F.softplus(self.rho).expand(means.shape)), dim=-1)
#         #return list(zip(self.mean.infer(input, samples), F.softplus(self.rho).repeat(samples)))

#     def all_losses(self):
#         return self.mean.all_losses()

#     @property
#     def var(self):
#         return F.softplus(self.rho)

class GaussLayer(nn.Module):
    def __init__(self, std_init: torch.Tensor, learn_var: bool = False):
        super().__init__()
        self.rho = torch.log(torch.exp(std_init) - 1)
        if learn_var:
            self.rho = nn.Parameter(self.rho)
        self.learn_var = learn_var

    def forward(self, input):
        return torch.stack((input, F.softplus(self.rho).expand(input.shape)), dim=-1)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return {
            "rho": self.rho
        }

    def load_state_dict(self, state):
        self.rho = state["rho"] 

    def all_losses(self):
        return self.mean.all_losses()

    @property
    def var(self):
        return F.softplus(self.rho)**2

def plot_losses(name, losses, ax):
    epochs = max([len(loss) for loss in losses])
    ax.set_xlabel("Epoch", fontsize=14)
    ax.set_xticks(np.arange(0, epochs + 1, epochs // 10))
    ax.set_ylabel("Training Loss", fontsize=14)
    if len(losses) > 1:
        for i, single_losses in enumerate(losses):
            ax.plot(np.arange(1, len(single_losses) + 1, 1), single_losses, label=f"{name} ({i})")
    else:
        ax.plot(np.arange(1, len(losses[0]) + 1, 1), losses[0], label=name)
    ax.legend(loc="upper right")