import torch
import torch.nn as nn
from torch.nn.utils.convert_parameters import parameters_to_vector, vector_to_parameters
from .network import generate_model

class RMSModule(nn.Module):
    def __init__(self, layers, gamma, noise, reg_scale):
        super().__init__()
        self.model = generate_model(layers)
        self.gamma = gamma
        self.noise = noise
        self.reg_scale = reg_scale
        self.losses = []
        self.anchor = [torch.normal(torch.zeros_like(params), gamma) for params in self.model.parameters()]
        self.infer_with_anchor = False

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return {
            "model": self.model.state_dict(destination, prefix, keep_vars),
            "anchor": self.anchor,
            "losses": self.losses
        }

    def load_state_dict(self, dict):
        self.model.load_state_dict(dict["model"])
        self.anchor = dict["anchor"]
        self.losses = dict["losses"]

    def train_model(self, epochs, loss_fn, optimizer_factory, loader, batch_size, device, scheduler_factory=None, report_every_epochs=1):
        self.model.to(device)
        self.model.train()
        optimizer = optimizer_factory(self.model.parameters())
        scheduler = scheduler_factory(optimizer) if scheduler_factory is not None else None

        for epoch in range(epochs):
            epoch_loss = torch.tensor(0, dtype=torch.float)
            for data, target in loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = self.model(data)
                data_loss = loss_fn(output, target)
                reg_loss = 1 / data.shape[0] * self.reg_loss()
                loss = data_loss + self.reg_scale * reg_loss
                loss.backward()
                optimizer.step()
                epoch_loss += loss.cpu().item()
            epoch_loss /= (len(loader) * batch_size)
            self.losses.append(epoch_loss.detach())

            if scheduler is not None:
                scheduler.step()

            if report_every_epochs > 0 and epoch % report_every_epochs == 0:
                print(f"Epoch {epoch}: loss {epoch_loss}")
        if report_every_epochs >= 0:
            print(f"Final loss {epoch_loss}")

    def infer(self, input, samples):
        if self.infer_with_anchor:
            old_params = parameters_to_vector(self.model.parameters())
            vector_to_parameters(parameters_to_vector(self.anchor), self.model.parameters())
            outputs = torch.stack([self.model(input) for _ in range(samples)])
            vector_to_parameters(old_params, self.model.parameters())
            return outputs

        self.model.eval()
        return torch.stack([self.model(input) for _ in range(samples)])

    def all_losses(self):
        return [self.losses]

    def reg_loss(self):
        return (self.noise/self.gamma)**2 * sum([((p - a)**2).sum() for (p, a) in zip(self.model.parameters(), self.anchor)])