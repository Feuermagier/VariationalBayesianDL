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
    def __init__(self, testloader, name, eval_fn, samples, device, fit_gaussian=True, cal_steps=10, target_mean=0, target_std=1):
        self.name = name

        mean_mse = 0
        mse_of_means = 0
        datapoints = 0
        log_likelihoods_per_sample = torch.zeros(samples)
        quantile_frequencies = torch.zeros(cal_steps)
        with torch.no_grad():
            for data, target in testloader:
                data = data.to(device)
                datapoints += len(data)
                torch.manual_seed(0) # Hoping that this produces deterministic outputs for correct LML calculation
                outputs = eval_fn(data, samples).cpu()
                # outputs = [samples,batch_size,out_dim,2 = mean + var]
                means = outputs[:,:,:,0] * target_std + target_mean
                vars = outputs[:,:,:,1] * target_std**2
                mean = means.mean(dim=0)
                target = target * target_std + target_mean
                for i in range(len(outputs)):
                    mean_mse += F.mse_loss(means[i], target, reduction="sum") / means.shape[0]
                    log_likelihoods_per_sample[i] += gauss_logprob(means[i], vars[i], target).sum()
                mse_of_means += F.mse_loss(mean, target, reduction="sum")
                quantile_frequencies += calc_quantile_frequencies(outputs, target, cal_steps, fit_gaussian)

        self.mean_mse = mean_mse / datapoints
        self.mse_of_means = mse_of_means / datapoints
        self.lml = -math.log(samples) + torch.logsumexp(log_likelihoods_per_sample, dim=0)
        self.average_lml = self.lml / datapoints
        self.observed_cdf = quantile_frequencies / len(testloader)
        self.quantile_ps = torch.linspace(0, 1, cal_steps)
        self.qce = (self.observed_cdf - self.quantile_ps).abs().mean()

def calc_quantile_frequencies(outputs, targets, quantile_steps, fit_gaussian):
    frequencies = torch.zeros(quantile_steps)
    quantile_ps = torch.linspace(0, 1, quantile_steps)
    samples = torch.distributions.Normal(outputs[:,:,:,0], outputs[:,:,:,1].sqrt()).sample()

    quantiles = torch.stack([torch.quantile(samples, p, dim=0) for p in quantile_ps])

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

def plot_table(title, results, filename=None):
    average_lmls = torch.tensor([[res.average_lml for res in reses] for reses in results])
    mean_mses = torch.tensor([[res.mean_mse for res in reses] for reses in results])
    mse_of_means = torch.tensor([[res.mse_of_means for res in reses] for reses in results])
    qces = torch.tensor([[res.qce for res in reses] for reses in results])
    divisior = math.sqrt(average_lmls.shape[1])
    texts = [[results[i][0].name, 
            f"{average_lmls[i].mean():.2f} ± {(average_lmls[i].std() / divisior):.2f}", 
            f"{mean_mses[i].mean():.3f} ± {(mean_mses[i].std() / divisior):.3f}", 
            f"{mse_of_means[i].mean():.3f} ± {(mse_of_means[i].std() / divisior):.3f}", 
            f"{qces[i].mean():.2f} ± {(qces[i].std() / divisior):.2f}"]
        for i in range(len(results))]
    cols = (title, "Avg LML", "Mean MSE", "MSE of Means", "QCE")
    table = tabulate(texts, headers=cols, tablefmt='orgtbl')
    print(table)

    if filename is not None:
        with open(filename, "w") as file:
            file.write(table)