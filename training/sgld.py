import torch
import torch.nn as nn
import numpy as np
from torch.optim.optimizer import Optimizer, required
import copy

from .network import generate_model

def sgld(lr):
    return lambda params: SGLD(params, lr=lr)

class SGLDModule(nn.Module):
    def __init__(self, layers, burnin_epochs, sample_interval):
        super().__init__()
        self.model = generate_model(layers)
        self.losses = []
        self.burnin_epochs = burnin_epochs
        self.sample_interval = sample_interval
        self.samples = []

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return {
            "model": self.model.state_dict(destination, prefix, keep_vars),
            "losses": self.losses,
            "samples": self.samples
        }

    def load_state_dict(self, dict):
        self.model.load_state_dict(dict["model"])
        self.losses = dict["losses"]
        self.samples = dict["samples"]

    def train_model(self, epochs, loss_fn, optimizer_factory, loader, batch_size, device, report_every_epochs=1):
        self.model.to(device)
        self.model.train()
        optimizer = optimizer_factory(self.model.parameters())

        for epoch in range(epochs):
            epoch_loss = torch.tensor(0, dtype=torch.float)
            for data, target in loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = loss_fn(output, target) * len(loader) * data.shape[-1] / 2
                loss.backward()
                optimizer.step()
                epoch_loss += loss.cpu()
            epoch_loss /= (len(loader) * batch_size)
            self.losses.append(epoch_loss.detach())

            if epoch >= self.burnin_epochs and (epoch - self.burnin_epochs) % self.sample_interval == 0:
                self.samples.append(copy.deepcopy(self.model.state_dict()))

            if report_every_epochs > 0 and epoch % report_every_epochs == 0:
                print(f"Epoch {epoch}: loss {epoch_loss}")
        if report_every_epochs >= 0:
            print(f"Final loss {epoch_loss}")

    def infer(self, input, samples):
        self.model.eval()
        backup = copy.deepcopy(self.model.state_dict())
        outputs = []
        for i in range(samples):
            self.model.load_state_dict(self.samples[i % len(self.samples)])
            outputs.append(self.model(input))
        self.model.load_state_dict(backup)
        return torch.stack(outputs)

    def all_losses(self):
        return [self.losses]

# Inspired by https://github.com/alisiahkoohi/Langevin-dynamics/blob/master/langevin_sampling/SGLD.py
class SGLD(torch.optim.SGD):
    def __init__(self, params, lr=required, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        super().__init__(params, lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)

    def step(self, closure=None):
        loss = super().step(closure)

        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                noise = torch.normal(torch.zeros_like(p), np.sqrt(lr))
                p.data.add_(noise)

        return loss