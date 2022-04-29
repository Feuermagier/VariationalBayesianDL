import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import itertools

def gauss_logprob(mean: torch.Tensor, variance: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    #var = torch.max(torch.tensor(1e-6), variance)
    return -((x - mean) ** 2) / (2 * variance) - torch.log(variance.sqrt()) - math.log(math.sqrt(2 * math.pi))

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
    def __init__(self, mean, var_init: torch.Tensor, learn_var: bool = False):
        super().__init__()
        self.mean = mean
        self.rho = torch.log(torch.exp(var_init) - 1).unsqueeze(0)
        if learn_var:
            self.rho = nn.Parameter(self.rho)
        self.learn_var = learn_var

    def forward(self, input):
        print(F.softplus(self.rho))
        return self.mean(input), F.softplus(self.rho).repeat(input.shape[0])

    def state_dict(self):
        return {
            "mean": self.mean.state_dict(),
            "rho": self.rho
        }

    def load_state_dict(self, state):
        self.mean.load_state_dict(state["mean"])
        self.rho = state["rho"]

    def train(self, epochs, optimizer_factory, loss_reduction, *args, **kwargs):
        if self.learn_var:
            optimizer_factory_ext = lambda p: optimizer_factory(list(p) + [self.rho])
        else:
            optimizer_factory_ext = optimizer_factory
        
        loss_fn = lambda output, target: F.gaussian_nll_loss(output, target, F.softplus(self.rho).repeat(output.shape[0]), reduction=loss_reduction)
        self.mean.train(epochs, loss_fn, optimizer_factory_ext, *args, **kwargs)
        

    def infer(self, input, samples):
        means = self.mean.infer(input, samples)
        return torch.stack((means, F.softplus(self.rho).expand(means.shape)), dim=-1)
        #return list(zip(self.mean.infer(input, samples), F.softplus(self.rho).repeat(samples)))

def map_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "tanh":
        return nn.Tanh()
    elif name == "logsoftmax":
        return nn.LogSoftmax(dim=1)
    else:
        raise ValueError(f"Unknown activation function {name}")

def generate_model(architecture, scale=1, linear_fn=lambda i, o: nn.Linear(i, o), conv_fn=lambda i, o, k: nn.Conv2d(i, o, k), dropout_p=0, dropout_fn=lambda p: nn.Dropout(p)):
    layers = []
    for i, (ty, size) in enumerate(architecture):
        if ty == "pool":
            layers.append(nn.MaxPool2d(size))
        elif ty == "flatten":
            layers.append(nn.Flatten())
        elif ty == "relu":
            layers.append(nn.ReLU())
        elif ty == "sigmoid":
            layers.append(nn.Sigmoid())
        elif ty == "logsoftmax":
            layers.append(nn.LogSoftmax(dim=1))
        elif ty == "fc":
            (in_features, out_features) = size
            in_features_scaled = in_features if i == 0 else int(in_features * scale)
            out_features_scaled = out_features if i == len(architecture) - 1 else int(out_features * scale)
            layers.append(linear_fn(int(in_features_scaled), int(out_features_scaled)))
            if i < len(architecture) - 1 and dropout_p > 0:
                layers.append(dropout_fn(dropout_p))
        elif ty == "conv":
            (in_channels, out_channels, kernel_size) = size
            layers.append(conv_fn(in_channels, out_channels, kernel_size))
        else:
            raise ValueError(f"Unknown layer type '{ty}'")
    model = nn.Sequential(*layers)
    print(f"Generated model: {model}")
    print(f"{sum([p.numel() for p in model.parameters() if p.requires_grad])} trainable parameters")
    return model

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