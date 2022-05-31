import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import math
import time
from .util import GaussianMixture
from .network import generate_model

class BBBModel(nn.Module):
    def __init__(self, layers):
        super().__init__()

        self.model = generate_model(layers)
        self.losses = []

    def state_dict(self):
        return {
            "model": self.model.state_dict(),
            "losses": self.losses
        }

    def load_state_dict(self, dict):
        self.model.load_state_dict(dict["model"])
        self.losses = dict["losses"]

    def train_model(self, epochs, data_loss_fn, optimizer_factory, loader, batch_size, device, mc_samples=5, kl_rescaling=1, report_every_epochs=1):
        self.model.to(device)
        self.model.train()
        optimizer = optimizer_factory(self.model.parameters())
        pi = kl_rescaling / len(loader)

        # kl_grads = []
        # data_grads = []
        for epoch in range(epochs):
            epoch_loss = torch.tensor(0, dtype=torch.float)
            for data, target in loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()

                kl_loss = torch.tensor(0, dtype=torch.float)
                data_loss = torch.tensor(0, dtype=torch.float)
                for _ in range(mc_samples):
                    output = self.model(data)
                    kl_loss += sum([getattr(layer, "kl", 0) for layer in self.model]) / data.shape[0]
                    data_loss += data_loss_fn(output, target)

                kl_loss /= mc_samples
                kl_loss *= pi
                data_loss /= mc_samples

                kl_loss.backward(retain_graph=True)
                #kl_grad = torch.cat([module.rho_grads() for module in self.model if hasattr(module, "rho_grads")])
                #kl_grads.append(kl_grad)

                data_loss.backward()
                #data_grad = torch.cat([module.rho_grads() for module in self.model if hasattr(module, "rho_grads")])
                #data_grad -= kl_grad
                #data_grads.append(data_grad)

                #nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), 10)
                optimizer.step()
                epoch_loss += pi * kl_loss + data_loss
            epoch_loss /= (len(loader) * batch_size)
            self.losses.append(epoch_loss.detach())
            if report_every_epochs > 0 and epoch % report_every_epochs == 0:
                print(f"Epoch {epoch}: loss {epoch_loss}")
        if report_every_epochs >= 0:
            print(f"Final loss {epoch_loss}")
        #return torch.stack(kl_grads), torch.stack(data_grads)

    def infer(self, input, samples):
        self.model.eval()
        return torch.stack([self.model(input) for _ in range(samples)])

    def all_losses(self):
        return [self.losses]

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