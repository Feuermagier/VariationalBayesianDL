from tabnanny import verbose
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import itertools

def gauss_logprob(mean: torch.Tensor, variance: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    #var = torch.max(torch.tensor(1e-6), variance)
    return -((x - mean) ** 2) / (2 * variance) - torch.log(variance.sqrt()) - math.log(math.sqrt(2 * math.pi))

def sgd(lr, momentum=0, weight_decay=0, nesterov=False):
    return lambda parameters: torch.optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)

def adam(lr, weight_decay=0):
    return lambda parameters: torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)

def nll_loss(output, target, eps: float = 1e-6,):
    mean = output[...,0]
    var = output[...,1]**2
    var = var.clamp(min=eps)
    #return F.gaussian_nll_loss(mean, target, std**2, reduction)
    # Custom implementation without any() to support functorch
    loss = 0.5 * (torch.log(var) + (mean - target)**2 / var)
    return loss.mean()

def lr_scheduler(milestones, gamma):
    return lambda opt: torch.optim.lr_scheduler.MultiStepLR(opt, milestones, gamma)

def wilson_scheduler(pretrain_epochs, lr_init, swag_lr=None):
    def wilson_schedule(epoch):
        t = (epoch) / pretrain_epochs
        lr_ratio = swag_lr / lr_init if swag_lr is not None else 0.01
        if t <= 0.5:
            factor = 1.0
        elif t <= 0.9:
            factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
        else:
            factor = lr_ratio
        return factor
    return lambda opt: torch.optim.lr_scheduler.LambdaLR(opt, wilson_schedule)

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
        if learn_var:
            self.std = nn.Parameter(std_init)
            self.std.should_sample = False
        else:
            self.register_buffer("std", std_init)
        self.learn_var = learn_var

    def forward(self, input):
        out = torch.stack((input, self.std.expand(input.shape)), dim=-1)
        return out

    @property
    def var(self):
        return self.std**2

def plot_losses(name, losses, ax, val_losses=[]):
    epochs = max([len(loss) for loss in losses])
    ax.set_xlabel("Epoch", fontsize=14)
    ax.set_xticks(np.arange(0, epochs + 1, epochs // 10 if epochs > 10 else 1))
    ax.set_ylabel("Training Loss", fontsize=14)
    if len(losses) > 1:
        for i, single_losses in enumerate(losses):
            ax.plot(np.arange(1, len(single_losses) + 1, 1), single_losses, label=f"{name} ({i})")
    else:
        ax.plot(np.arange(1, len(losses[0]) + 1, 1), losses[0], label=name)
    
    if len(val_losses) > 0:
        ax.plot(np.arange(1, len(val_losses) + 1, 1), val_losses, label=name + " (Validation)")

    ax.legend(loc="upper right")


class EarlyStopper:
    def __init__(self, evaluator, interval, delta, patience):
        self.evaluator = evaluator
        self.interval = interval
        self.delta = delta
        self.patience = patience

        self.losses = []
        self.best_loss = float("inf")
        self.epochs_since_best = 0

    def should_stop(self, model, epoch):
        if epoch % self.interval != 0:
            return False

        with torch.no_grad():
            loss = self.evaluator(model)
            self.losses.append(loss)

            if loss < self.best_loss - self.delta:
                self.best_loss = loss
                self.epochs_since_best = 0
            else:
                self.epochs_since_best += 1
            #print(f"val loss {loss}")
            #print(f"patience {self.epochs_since_best}")

            if self.epochs_since_best > self.patience:
                print(f"Stopping early")
                return True
            else:
                return False
