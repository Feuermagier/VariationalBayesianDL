import torch
import torch.nn as nn
import copy

class SWAGWrapper:
    def __init__(self, model: nn.Module, update_freq: int, K: int):
        self.model = model
        self.weights = torch.nn.utils.convert_parameters.parameters_to_vector(self.model.parameters())
        self.sq_weights = self.weights**2
        self.updates = 0
        self.deviations = torch.zeros((self.weights.shape[0], K))
        self.update_freq = update_freq
        self.K = K

    @property
    def mean(self):
        return self.weights
    
    @property
    def covariance(self):
        diag = torch.diag(self.sq_weights - self.weights**2 + 1e-8) # Adding 1e-8 for numerical stability
        low_rank = self.deviations @ torch.transpose(self.deviations, 0, 1) / (self.K - 1)
        return 0.5 * diag + 0.5 * low_rank

    def update(self, epoch):
        if epoch % self.update_freq == 0:
            self.updates += 1
            params = torch.nn.utils.convert_parameters.parameters_to_vector(self.model.parameters())
            self.weights = (self.updates * self.weights + params) / (self.updates + 1)
            self.sq_weights = (self.updates * self.sq_weights + params**2) / (self.updates + 1)
            self.deviations = torch.roll(self.deviations, -1, 1)
            self.deviations[:,-1] = params - self.weights

    def sample(self, model: nn.Module, input: torch.Tensor):
        dist = torch.distributions.MultivariateNormal(self.mean, self.covariance)
        torch.nn.utils.convert_parameters.vector_to_parameters(dist.sample(), model.parameters())
        return model(input)