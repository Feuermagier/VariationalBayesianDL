import torch

# Weighted sum of two gaussian distributions
class GaussianMixture:
    def __init__(self, pi, sigma1, sigma2):
        self.pi, self.sigma1, self.sigma2 = pi, sigma1, sigma2
        self.gaussian1 = torch.distributions.Normal(0, sigma1)
        self.gaussian2 = torch.distributions.Normal(0, sigma2)

    def log_prob(self, value):
        p1 = torch.exp(self.gaussian1.log_prob(value))
        p2 = torch.exp(self.gaussian2.log_prob(value))
        return torch.log((self.pi * p1 + (1 - self.pi) * p2)).sum()