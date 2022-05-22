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

    def train_model(self, epochs, report_every_epochs=1, attempts=5):
        self.likelihood.train()

        best_loss = 100000
        for i in range(attempts):
            if report_every_epochs >= 0:
                print(f"Training attempt {i}")

            indices = torch.randperm(len(self.xs))

            gp = _GPModel(self.likelihood, self.xs[indices], self.ys[indices])
            gp.train()
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, gp)
            optimizer = torch.optim.Adam(gp.parameters(), lr=0.1)

            losses = []
            for epoch in range(epochs):
                optimizer.zero_grad()
                output = gp(self.xs[indices])
                loss = -mll(output, self.ys[indices])
                loss.backward()
                optimizer.step()
                losses.append(loss.detach())
                if report_every_epochs > 0 and epoch % report_every_epochs == 0:
                    print(f"Epoch {epoch}: loss {loss}")
            if loss < best_loss:
                best_loss = loss
                self.gp = gp
                self.mll = mll
                self.losses = losses

            if report_every_epochs >= 0:
                print(f"Final loss {loss}")

    def infer(self, input, samples):
        self.gp.eval()
        self.likelihood.eval()
        dist = self.gp(input.squeeze(-1))
        outputs = torch.stack([dist.sample().unsqueeze(-1) for _ in range(samples)])
        return torch.stack((outputs, self.noise.expand(outputs.shape)), -1)

    def all_losses(self):
        return [self.losses]