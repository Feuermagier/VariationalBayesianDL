import torch
import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt
import torch.nn.functional as F
import math
from .util import gauss_logprob

class RegressionToyDataset(torch.utils.data.Dataset):
    def __init__(self, min: float, max: float, sample_count: int, normalize: bool, noise: float):
        super().__init__()
        self.min, self.max = min, max

        if normalize:
            self.x_norm = 1 / np.abs(max)
            self.y_norm = 1 / np.abs(self.eval_value(torch.tensor(max)))
        else:
            self.x_norm = 1
            self.y_norm = 1

        xs, ys = _sample_from_fn(self.eval, min, max, sample_count, noise)
        xs, ys = torch.unsqueeze(xs, -1) * self.x_norm, torch.unsqueeze(ys, -1) * self.y_norm
        self.samples = [(x, y) for x, y in zip(xs, ys)] # Just using zip() doesn't work for some reason

    def eval_value(self, value):
        if isinstance(value, torch.Tensor):
            return self.eval(value, torch.zeros(value.shape))
        else:
            return self.eval(torch.tensor(value), torch.tensor(0))

    def __iter__(self):
        return iter(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, key):
        return self.samples[key]

    def plot(self, name, eval_fn, extra_range=0.01, plot_sigma=False):
        plt.title(name)
        extra = (self.max - self.min) * extra_range
        min, max = self.min - extra, self.max + extra
        #plt.xlim(min, max)
        t = torch.linspace(min, max, 50)
        y = self.eval_value(t)

        with torch.no_grad():
            outputs = eval_fn(torch.unsqueeze(t * self.x_norm, -1))
        
        means = torch.empty((len(outputs), t.shape[0]))
        mses = torch.empty((len(outputs)))
        log_likelihoods = torch.empty((len(outputs), t.shape[0]))
        for i, (mean, variance) in enumerate(outputs):
            mean = torch.squeeze(mean, -1).detach() / self.y_norm
            variance = torch.squeeze(variance, -1).detach() / self.y_norm**2
            means[i] = mean
            mse = F.mse_loss(mean, y)
            mses[i] = mse
            log_likelihoods[i] = gauss_logprob(mean, variance, y)

            plt.plot(t, mean, color="red")

            if plot_sigma:
                higher_bound = mean + 3 * torch.sqrt(variance)
                lower_bound = mean - 3 * torch.sqrt(variance)
                plt.fill_between(t, lower_bound, higher_bound, color="lightgrey")

        marginal_log_likelihood = -math.log(len(outputs)) + torch.logsumexp(log_likelihoods, dim=0)

        text = f"{len(outputs)} weight sample(s)\n" \
            + f"Average marginal log likelihood: {marginal_log_likelihood.mean()}\n" \
            + f"MSE of average: {F.mse_loss(means.mean(dim=0), y)}\n" \
            + f"Average MSE: {mses.mean(dim=0)}\n" \
            + f"Minimal MSE: {mses.min(dim=0)[0]}"
        plt.figtext(0.5, 0.01, text, ha="center", va="top", fontsize=12)

        plt.plot(t, y, color="blue") # Actual function

        xs, ys = zip(*(((x / self.x_norm).numpy(), (y / self.y_norm).numpy()) for (x, y) in self))
        plt.scatter(xs, ys, s=4, color="blue", zorder=10)

def _sample_from_fn(function, min, max, sample_count, noise_sigma):
    xs = (max - min) * torch.rand(sample_count) + min
    noise = torch.normal(mean=torch.zeros(sample_count), std=torch.full((sample_count,), noise_sigma))
    ys = function(xs, noise)
    return xs.float(), ys.float()

# See arXiv:1502.05336 (also used in arXiv:1612.01474)
class CubicToyDataset(RegressionToyDataset):
    def __init__(self, min: float = -4, max: float = 4, sample_count: int = 20, normalize: bool = True, noise: float = 3, offset: float = 0):
        self.offset = offset
        super().__init__(min, max, sample_count, normalize, noise)

    def eval(self, value, noise):
        return value**3 + self.offset + noise

class TrigonometricToyDataset(RegressionToyDataset):
    def __init__(self, min: float = 0, max: float = 0.5, sample_count: int = 20, normalize: bool = False, noise: float = 0.02):
        super().__init__(min, max, sample_count, normalize, noise)

    def eval(self, value, noise):
        return value + 0.3*torch.sin(2*torch.pi*(value + noise)) + 0.3*torch.sin(4*torch.pi*(value + noise)) + noise


class ClassificationToyDataset(torch.utils.data.Dataset):
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels

    def __iter__(self):
        return zip(self.samples, self.labels)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, key):
        return self.samples[key], self.labels[key]

class TwoMoonsDataset(ClassificationToyDataset):
    def __init__(self, samples: int = 100, noise: float = 0.1, seed: int = None):
        data, labels = sklearn.datasets.make_moons(samples, noise=noise, random_state=seed)
        super().__init__(torch.tensor(data, dtype=torch.float), torch.tensor(labels, dtype=torch.float).unsqueeze(-1))