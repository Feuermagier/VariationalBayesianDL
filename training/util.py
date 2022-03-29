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
        self.log_pi, self.sigma1, self.sigma2 = torch.log(torch.tensor(pi)), sigma1, sigma2
        self.gaussian1 = torch.distributions.Normal(0, sigma1)
        self.gaussian2 = torch.distributions.Normal(0, sigma2)

    def log_prob(self, value):
        #p1 = torch.exp(self.gaussian1.log_prob(value)) + 1e-6
        #p2 = torch.exp(self.gaussian2.log_prob(value)) + 1e-6
        #return torch.log((self.pi * p1 + (1 - self.pi) * p2)).sum()
        return torch.logaddexp(self.log_pi + self.gaussian1.log_prob(value), self.log_pi + self.gaussian2.log_prob(value))

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

def map_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "tanh":
        return nn.Tanh()
    else:
        raise ValueError(f"Unknown activation function {name}")

def generate_model(architecture, activation, out_activation, scale=1, linear_fn=lambda i, o: nn.Linear(i, o), dropout_p=0):
    layers = []
    for i, (in_features, out_features) in enumerate(architecture):
        in_features_scaled = in_features if i == 0 else int(in_features * scale)
        out_features_scaled = out_features if i == len(architecture) - 1 else int(out_features * scale)
        layers.append(linear_fn(int(in_features_scaled), int(out_features_scaled)))
        if i < len(architecture) - 1:
            layers.append(map_activation(activation))
            if dropout_p > 0:
                layers.append(nn.Dropout(dropout_p))
        elif out_activation is not None:
            layers.append(map_activation(out_activation))
    model = nn.Sequential(*layers)
    print(f"Generated model: {model}")
    return model