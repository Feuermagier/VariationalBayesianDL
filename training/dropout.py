import torch
import torch.nn as nn
import torch.nn.functional as F

class FixableDropout(nn.Module):
    def __init__(self, p, freeze_on_eval=True):
        super().__init__()
        self.p = torch.tensor(p)
        self.freeze_on_eval = freeze_on_eval

    def forward(self, x):
        if not self.training and self.freeze_on_eval:
            mask = (1 - self.p).expand(x.shape[1:])
            mask = torch.bernoulli(mask).to(x.device)
            return x * mask
        else:
            return F.dropout(x, self.p)
        