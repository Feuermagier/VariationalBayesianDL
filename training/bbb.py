# See https://github.com/nitarshan/bayes-by-backprop/blob/master/Weight%20Uncertainty%20in%20Neural%20Networks.ipynb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from .util import GaussianMixture

# Multidimensional gaussian distribution; for sampling weights and biases of a nn layer. Uses the reparametrization trick
class GaussianParameters:
    def __init__(self, mu: torch.Tensor, rho: torch.Tensor, device: torch.device):
        self.mu, self.rho = mu, rho
        self.distribution = torch.distributions.Normal(0, 1)
        self.device = device

    @property
    def sigma(self):
        return torch.log1p(torch.exp(self.rho))

    def sample(self):
        epsilon = self.distribution.sample(self.rho.size()).to(self.device)
        return self.mu + self.sigma * epsilon

    def log_prob(self, value):
        self.distribution.log_prob(value)
        return (-np.log(np.sqrt(2 * np.pi)) - torch.log(self.sigma) - ((value - self.mu) ** 2) / (2 * self.sigma ** 2)).sum()


# Single linear layer of a BBB network. In essence this is a normal fc layer, but the weights and biases
# are drawn from a gaussian distribution and the corresponding means and variances are learned
#
# This class also stores the prior and variational posterior of the weights and biases of the last
# forward pass through the layer
class BayesianLinearLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, weight_prior: GaussianMixture, bias_prior: GaussianMixture, device: torch.device, weight_draw: str = "sample", deterministic_eval: bool = False):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.weight_prior, self.bias_prior = weight_prior, bias_prior
        self.log_prior, self.log_posterior = 0, 0
        self.weight_draw = weight_draw
        self.deterministic_eval = deterministic_eval

        # Weights
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.2, 0.2))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-5, -4))
        self.weight = GaussianParameters(self.weight_mu, self.weight_rho, device)

        # Biases
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-5, -4))
        self.bias = GaussianParameters(self.bias_mu, self.bias_rho, device)

    def forward(self, input: torch.Tensor):
        if not self.training and self.deterministic_eval:
            weight = self.weight.mu
            bias = self.bias.mu
            return F.linear(input, weight, bias)
        elif self.weight_draw == "minibatch":
            weight = self.weight.sample()
            bias = self.bias.sample()
            self.log_prior = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias)
            self.log_posterior = self.weight.log_prob(weight) + self.bias.log_prob(bias)
            return F.linear(input, weight, bias)
        elif self.weight_draw == "sample":
            output = torch.zeros((input.shape[0], self.out_features))
            self.log_prior = 0
            self.log_posterior = 0
            for i, sample in enumerate(input):
                weight = self.weight.sample()
                bias = self.bias.sample()
                self.log_prior += self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias)
                self.log_posterior += self.weight.log_prob(weight) + self.bias.log_prob(bias)
                output[i] = weight @ sample + bias
            return output
        else:
            raise ValueError("Invalid value of weight_draw")

