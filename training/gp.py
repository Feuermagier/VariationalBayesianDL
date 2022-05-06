import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from training import util
import gpytorch

class _GPModel(gpytorch.models.ExactGP):
    def __init__(self, likelihood, xs, ys):
        super().__init__(xs, ys, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class GaussianProcess:
    def __init__(self, xs, ys, noise):
        self.xs, self.ys = torch.squeeze(xs, -1), torch.squeeze(ys, -1)
        self.noise = noise

        self.likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=noise.expand(xs.shape[0]))
        #self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.gp = _GPModel(self.likelihood, self.xs, self.ys)
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.gp)

    def train_model(self, epochs, report_every_epochs=1):
        self.gp.train()
        self.likelihood.train()
        optimizer = torch.optim.Adam(self.gp.parameters(), lr=0.1)

        losses = []
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = self.gp(self.xs)
            loss = -self.mll(output, self.ys)
            loss.backward()
            optimizer.step()
            losses.append(loss)
            if report_every_epochs > 0 and epoch % report_every_epochs == 0:
                print(f"Epoch {epoch}: loss {loss}")
        if report_every_epochs >= 0:
            print(f"Final loss {loss}")
        self.losses = losses

    def infer(self, input, samples):
        self.gp.eval()
        self.likelihood.eval()
        dist = self.gp(input.squeeze(-1))
        outputs = torch.stack([dist.sample().unsqueeze(-1) for _ in range(samples)])
        return torch.stack((outputs, self.noise.expand(outputs.shape)), -1)

    def all_losses(self):
        return [self.losses]