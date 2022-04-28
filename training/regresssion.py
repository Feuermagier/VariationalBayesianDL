from multiprocessing import reduction
import torch
import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch.nn.functional as F
from scipy.stats import wasserstein_distance
import math
from tabulate import tabulate
from .util import gauss_logprob

class RegressionResults:
    def __init__(self, testloader, name, eval_fn, samples, variance, device, fit_gaussian=True, cal_steps=10):
        self.name = name

        mean_mse = 0
        mse_of_means = 0
        datapoints = 0
        log_likelihoods_per_sample = torch.zeros(samples)
        quantile_frequencies = torch.zeros(cal_steps)
        with torch.no_grad():
            for data, target in testloader:
                data, target = data.to(device), target.to(device)
                datapoints += len(data)
                torch.manual_seed(0) # Hoping that this produces deterministic outputs for correct LML calculation
                outputs = torch.stack(eval_fn(data, samples)).cpu()
                mean = outputs.mean(dim=0)
                for output in outputs:
                    mean_mse += F.mse_loss(output, target, reduction="sum")
                mean_mse /= outputs.shape[0]
                mse_of_means += F.mse_loss(mean, target, reduction="sum")
                quantile_frequencies += calc_quantile_frequencies(outputs, target, cal_steps, fit_gaussian)
                log_likelihoods_per_sample += gauss_logprob(mean, variance, target).sum()

        self.mean_mse = mean_mse / datapoints
        self.mse_of_means = mse_of_means / datapoints
        self.lml = -math.log(samples) + torch.logsumexp(log_likelihoods_per_sample, dim=0)
        self.observed_cdf = quantile_frequencies / len(testloader)
        self.quantile_ps = torch.linspace(0, 1, cal_steps)
        self.qce = (self.observed_cdf - self.quantile_ps).abs().mean()

def calc_quantile_frequencies(outputs, targets, quantile_steps, fit_gaussian):
    frequencies = torch.zeros(quantile_steps)
    quantile_ps = torch.linspace(0, 1, quantile_steps)

    if fit_gaussian:
        dist = torch.distributions.Normal(outputs.mean(dim=0), outputs.std(dim=0))
        quantiles = dist.icdf(quantile_ps.expand((len(targets), quantile_steps)).T.unsqueeze(-1))
    else:
        quantiles = torch.stack([torch.quantile(outputs, p, dim=0) for p in quantile_ps])

    for i, quantile in enumerate(quantiles):
        frequencies[i] = (targets <= quantile).sum()

    return frequencies / len(targets)

def plot_calibration(title, results, ax):
    ax.set_xlabel("Expected Confidence Level", fontsize=14)
    ax.set_ylabel("Observed Confidence Level", fontsize=14)
    ax.plot([0, 1], [0,1])
    ax.plot(results.quantile_ps, results.observed_cdf, "o-")
    text = f"{title}\nQCE: {results.qce:.3f}"
    ax.text(0.08, 0.9, text, 
        transform=ax.transAxes, fontsize=14, verticalalignment="top", 
        bbox={"boxstyle": "square,pad=0.5", "facecolor": "white"})

def plot_table(results):
    texts = [[res.name, f"{res.lml:.3f}", f"{res.mean_mse:.3f}", f"{res.mse_of_means:.3f}", f"{res.qce:.3f}"] for res in results]
    cols = ("Method", "LML", "Mean MSE", "MSE of Means", "QCE")
    print(tabulate(texts, headers=cols, tablefmt='orgtbl'))