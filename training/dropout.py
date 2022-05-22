import torch
import torch.nn as nn

class FixableDropout(nn.Module):
    def __init__(self, p, freeze_on_eval=True):
        super().__init__()
        self.p = torch.tensor(1 - p)
        self.freeze_on_eval = freeze_on_eval

    def forward(self, x):
        if not self.training and self.freeze_on_eval:
            mask = self.p.expand(x.shape[1:])
        elif self.freeze_on_eval:
            mask = self.p.expand(x.shape)
        mask = torch.bernoulli(mask).to(x.device)
        return x * mask