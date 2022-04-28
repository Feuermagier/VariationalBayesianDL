from pickletools import optimize
import torch
import torch.nn as nn
from .util import generate_model

class PointPredictor(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.model = generate_model(layers)
        self.losses = []

    def state_dict(self, ):
        return {
            "model": self.model.state_dict(),
            "losses": self.losses
        }

    def load_state_dict(self, dict):
        self.model.load_state_dict(dict["model"])
        self.losses = dict["losses"]

    def train(self, epochs, loss_fn, optimizer_factory, loader, batch_size, device, report_every_epochs=1):
        self.model.to(device)
        self.model.train()
        optimizer = optimizer_factory(self.model.parameters())

        for epoch in range(epochs):
            epoch_loss = torch.tensor(0, dtype=torch.float)
            for data, target in loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = loss_fn(output, target)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.cpu()
            epoch_loss /= (len(loader) * batch_size)
            self.losses.append(epoch_loss.detach())
            if report_every_epochs > 0 and epoch % report_every_epochs == 0:
                print(f"Epoch {epoch}: loss {epoch_loss}")
        if report_every_epochs >= 0:
            print(f"Final loss {epoch_loss}")

    def infer(self, input, samples):
        return [self.model(input) for _ in range(samples)]

    def all_losses(self):
        return [self.losses]