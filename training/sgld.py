import torch
import torch.nn as nn
import numpy as np
from torch.optim.optimizer import Optimizer, required
import copy

from .network import generate_model

def sgld(lr, temperature=1):
    return lambda params: SGLD(params, lr=lr, temperature=temperature)

def psgld(lr, temperature=1):
    return lambda params: PSGLD(params, lr=lr, temperature=temperature)

class SGLDModule(nn.Module):
    def __init__(self, layers, burnin_epochs, sample_interval, chains=1):
        super().__init__()
        self.chains = chains
        self.layers = layers
        self.losses = []
        self.burnin_epochs = burnin_epochs
        self.sample_interval = sample_interval
        self.samples = []

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return {
            "losses": self.losses,
            "samples": self.samples
        }

    def load_state_dict(self, dict):
        self.losses = dict["losses"]
        self.samples = dict["samples"]

    def train_model(self, epochs, loss_fn, optimizer_factory, loader, batch_size, device, report_every_epochs=1):
        for i in range(self.chains):
            model = generate_model(self.layers)
            if report_every_epochs >= 0:
                print(f"Running chain {i}")

            model.to(device)
            model.train()
            parameters = []
            for layer in model:
                if type(layer).__name__ == "GaussLayer":
                    parameters.append({"params": layer.parameters(), "noise": False})
                else:
                    parameters.append({"params": layer.parameters(), "noise": True})
            optimizer = optimizer_factory(parameters)

            for epoch in range(epochs):
                epoch_loss = torch.tensor(0, dtype=torch.float)
                for data, target in loader:
                    data, target = data.to(device), target.to(device)
                    optimizer.zero_grad()
                    output = model(data)
                    loss = loss_fn(output, target) * len(loader) * data.shape[-1]
                    loss.backward()
                    optimizer.step(burnin=False)
                    epoch_loss += loss.cpu().item()
                    #print(self.model[-1].var)
                epoch_loss /= (len(loader) * batch_size)
                self.losses.append(epoch_loss.detach())

                if epoch == self.burnin_epochs and report_every_epochs >= 0:
                    print(f"SGLD: Burnin completed in epoch {epoch}; now collecting posterior samples")

                if epoch >= self.burnin_epochs and (epoch - self.burnin_epochs) % self.sample_interval == 0:
                    self.samples.append(copy.deepcopy(model.state_dict()))

                if report_every_epochs > 0 and epoch % report_every_epochs == 0:
                    print(f"Epoch {epoch}: loss {epoch_loss}")
            if report_every_epochs >= 0:
                print(f"SGLD: Collected {len(self.samples)} posterior samples")

    def infer(self, input, samples):
        model = generate_model(self.layers)
        outputs = []
        for i in range(samples):
            model.load_state_dict(self.samples[i % len(self.samples)])
            outputs.append(model(input))
        return torch.stack(outputs)

    def all_losses(self):
        return [self.losses]

# Inspired by https://github.com/alisiahkoohi/Langevin-dynamics/blob/master/langevin_sampling/SGLD.py
class SGLD(torch.optim.SGD):
    def __init__(self, params, lr=required, temperature=1, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        super().__init__(params, lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)
        self.temperature = temperature

    def step(self, closure=None, burnin=False):
        loss = super().step(closure)

        if burnin:
            return loss

        for group in self.param_groups:
            lr = group["lr"]
            if "noise" in group and group["noise"] is False:
                continue

            for p in group["params"]:
                if p.grad is None:
                    continue
                noise = torch.normal(torch.zeros_like(p), np.sqrt(2 * lr * self.temperature))
                p.data.add_(noise)

        return loss

class PSGLD(torch.optim.RMSprop):
    def __init__(self, params, lr=required, temperature=1, alpha=0.99, eps=1e-05, weight_decay=0, momentum=0, centered=False):
        super().__init__(params, lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, momentum=momentum, centered=centered)
        self.temperature = temperature

    def step(self, closure=None, burnin=False):
        loss = super().step(closure)

        if burnin:
            return loss

        for group in self.param_groups:
            if "noise" in group and group["noise"] is False:
                continue
            lr = group["lr"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]

                square_avg = state["square_avg"]
                avg = square_avg.sqrt().add_(eps)
                noise = torch.normal(torch.zeros_like(p), np.sqrt(lr * self.temperature)) / avg
                p.data.add_(noise)

        return loss