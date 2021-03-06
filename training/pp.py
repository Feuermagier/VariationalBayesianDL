import torch
import torch.nn as nn
from .network import generate_model
from torchviz import make_dot

class MAP(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.model = generate_model(layers)
        self.losses = []

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return {
            "model": self.model.state_dict(destination, prefix, keep_vars),
            "losses": self.losses
        }

    def load_state_dict(self, dict):
        self.model.load_state_dict(dict["model"])
        self.losses = dict["losses"]

    def train_model(self, epochs, loss_fn, optimizer_factory, loader, batch_size, device, mc_samples=1, scheduler_factory=None, report_every_epochs=1, early_stopping=None):
        self.model.to(device)
        self.model.train()
        optimizer = optimizer_factory(self.model.parameters())
        scheduler = scheduler_factory(optimizer) if scheduler_factory is not None else None

        for epoch in range(epochs):
            epoch_loss = torch.tensor(0, dtype=torch.float)
            for data, target in loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()

                loss = torch.tensor(0.0, device=data.device)
                for _ in range(mc_samples):
                    output = self.model(data)
                    loss += loss_fn(output, target)
                loss /= mc_samples
                loss.backward()
                optimizer.step()
                epoch_loss += loss.cpu().item()
                #return make_dot(loss, params=dict(self.model.named_parameters()))
            epoch_loss /= len(loader)
            self.losses.append(epoch_loss.detach())

            if scheduler is not None:
                scheduler.step()

            if report_every_epochs > 0 and epoch % report_every_epochs == 0:
                print(f"Epoch {epoch}: loss {epoch_loss}")

            if early_stopping is not None and early_stopping.should_stop(self.infer, epoch):
                break

        if report_every_epochs >= 0:
            print(f"Final loss {epoch_loss}")

    def forward(self, input, samples=1):
        return self.infer(input, samples)

    def infer(self, input, samples):
        self.model.eval()
        return torch.stack([self.model(input) for _ in range(samples)])

    def all_losses(self):
        return [self.losses]