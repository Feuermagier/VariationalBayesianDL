import torch
import torch.nn as nn
import numpy as np
from torch.optim.optimizer import Optimizer, required

class iVON(Optimizer):
    def __init__(self, params, lr, N, betas, prior_prec, damping=1, tempering=1, augmentation=1):
        defaults = {
            "lr": lr,
            "betas": betas,
            "prior_prec": prior_prec,
            "damping": damping,
            "tempering": tempering,
            "augmentation": augmentation
        }
        super().__init__(params, defaults)
        self.N = N

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            delta = group["tempering"] * group["prior_prec"] / self.N
            beta_1, beta_2 = group["betas"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["scale"] = 1 #TODO
                    state["exp_avg"] = 0
                
                grad_mu = p.grad + delta * p.data
                state["m"].mul_(beta_1).add_(grad_mu.mul_(1 - beta_1))
                grad_s = delta - state["scale"] + (self.N * state["scale"] * ())
                

                    

        return loss
        