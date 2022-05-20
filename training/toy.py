import torch
import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch.nn.functional as F
from scipy.stats import wasserstein_distance
import math
from .util import gauss_logprob

class RegressionToyDataset:
    # data_areas = [(min, max, datapoints)]; must be sorted
    def __init__(self, data_areas, noise):
        super().__init__()

        # Generate datapoints
        xs, ys = [], []
        for (min, max, datapoints) in data_areas:
            area_xs = min + (max - min) * torch.rand(datapoints)
            area_ys = self.eval(area_xs, torch.normal(torch.zeros(datapoints), noise))
            xs.append(area_xs)
            ys.append(area_ys)
        xs, ys = torch.cat(xs).unsqueeze(-1), torch.cat(ys).unsqueeze(-1)

        # Shuffle
        permutation = torch.randperm(xs.shape[0])
        self.xs, self.ys = xs[permutation], ys[permutation]

        # Normalize
        self.x_mean = torch.mean(self.xs, dim=0)
        self.x_std = torch.std(self.xs, dim=0)
        self.y_mean = torch.mean(self.ys, dim=0)
        self.y_std = torch.std(self.ys, dim=0)
        # self.x_mean = 0
        # self.x_std = 1
        # self.y_mean = 0
        # self.y_std = 1

        self.normalized_xs = (xs - self.x_mean) / self.x_std
        self.normalized_ys = (ys - self.y_mean) / self.y_std

        self.trainset = torch.utils.data.TensorDataset(self.normalized_xs, self.normalized_ys)

    def generate_testset(self, min, max, datapoints, noise):
        xs = torch.linspace(min, max, datapoints)
        ys = self.eval(xs, torch.normal(torch.zeros(datapoints), noise))
        return torch.utils.data.TensorDataset((xs.unsqueeze(-1) - self.x_mean ) / self.x_std, (ys.unsqueeze(-1) - self.y_mean) / self.y_std)

    def plot(self, name, eval_fn, gp_eval, variance, extra_range=0.01, plot_sigma=False, alpha=1, samples=100, plot_lml_trend=None, gp_lml=None):
        fig = plt.figure(figsize=(15, 6))
        fig.suptitle(name, fontsize=16)

        # Data points + predictions
        data_ax = fig.add_subplot(1, 2, 1)

        t = self.generate_eval_range(extra_range)
        y = self.eval_value(t)
        #plt.ylim(-0.3, 0.9)

        outputs = self.generate_samples(eval_fn, samples, t)
        
        # Plot samples and calculate MSEs
        means = torch.empty((samples, t.shape[0]))
        mses = torch.empty(samples)
        for i, mean in enumerate(outputs):
            mean = torch.squeeze(mean, -1).detach() / self.y_norm
            #variance = torch.squeeze(variance, -1).detach() / self.y_norm**2
            means[i] = mean
            mse = F.mse_loss(mean, y)
            mses[i] = mse

            data_ax.plot(t, mean, color="red", alpha=alpha)

            if plot_sigma:
                higher_bound = mean + 3 * torch.sqrt(variance)
                lower_bound = mean - 3 * torch.sqrt(variance)
                data_ax.fill_between(t, lower_bound, higher_bound, color="lightgrey")

        total_mean = means.mean(dim=0)
        log_marginal_likelihood = calculate_lml_gaussian(y, outputs, variance)

        wasserstein_dists = []
        ref_means = gp_eval(t, samples)
        ref_means = torch.stack(ref_means)
        for mean, ref_mean in zip(means.T, ref_means.T):
            wasserstein_dists.append(wasserstein_distance(mean, ref_mean))
        wasserstein_dists = torch.tensor(wasserstein_dists)

        text = f"{samples} parameter sample(s)\n" \
            + f"Avg W(Model,GP): {wasserstein_dists.mean():.5f}\n" \
            + f"LML: {log_marginal_likelihood:.5f}\n" \
            + f"MSE of average: {F.mse_loss(total_mean, y):.5f}\n" \
            + f"Average MSE: {mses.mean(dim=0):.5f}\n" \
            + f"Minimal MSE: {mses.min(dim=0)[0]:.5f}"

        data_ax.text(0.5, -0.1, text, ha="center", va="top", fontsize=14, transform=data_ax.transAxes)

        data_ax.plot(t, y, color="blue") # Actual function

        xs, ys = zip(*(((x / self.x_norm).numpy(), (y / self.y_norm).numpy()) for (x, y) in self))
        data_ax.scatter(xs, ys, s=4, color="blue", zorder=10)

        if plot_lml_trend is not None:
            (lml_max, lml_steps) = plot_lml_trend
            lml_sample_counts = torch.linspace(1, lml_max, steps=lml_steps, dtype=torch.int)
            lmls = []
            for s in lml_sample_counts:
                lml_samples = []
                for _ in range(10):
                    outputs = self.generate_samples(eval_fn, s, t)
                    lml_samples.append(calculate_lml_gaussian(y, outputs, variance))
                lmls.append(sum(lml_samples) / len(lml_samples))
            lml_ax = fig.add_subplot(1, 2, 2)
            if gp_lml is not None:
                true_lml = gp_lml(t, y)
                lml_ax.axhline(y=true_lml, color="grey", linestyle="--")
                lml_ax.text(0.5, 0.3, f"True LML: {true_lml:.2f}", 
                    transform=lml_ax.transAxes, fontsize=14, verticalalignment="top", 
                    bbox={"boxstyle": "square,pad=0.5", "facecolor": "white"})
            lml_ax.set_xlabel("Parameter Samples", fontsize=14)
            lml_ax.set_ylabel("LML", fontsize=14)
            lml_ax.plot(lml_sample_counts, lmls)

    def plot_dataset(self, min, max, axis, dataset=None, zorder_offset=0):
        #plt.xlim(min, max)
        t = torch.linspace(min, max, 100)
        y = self.eval(t, torch.zeros(100))
        axis.plot(t, y, color="blue")
        if dataset is None:
            axis.scatter(self.xs, self.ys, s=4, color="blue")
        else:
            axis.scatter(dataset.tensors[0] * self.x_std + self.x_mean, dataset.tensors[1] * self.y_std + self.y_mean, s=4, color="blue", zorder=1+zorder_offset)

    def plot_predictions(self, min, max, eval_fn, samples, axis, dataset=None, alpha=1):
        self.plot_dataset(min, max, axis, dataset)
        t = torch.linspace(min, max, 100)
        with torch.no_grad():
            y = eval_fn((t.unsqueeze(-1) - self.x_mean) / self.x_std, samples) * self.y_std + self.y_mean
        for sample in y:
            axis.plot(t, sample[:,:,0], color="red", alpha=alpha, zorder=5 if alpha > 0.5 else 0)

def calculate_lml_gaussian(target, samples, var):
    assert len(samples) > 0
    log_likelihoods = torch.empty(len(samples))
    for i, mean in enumerate(samples):
        log_likelihoods[i] = gauss_logprob(mean, var, target).sum()
    return -math.log(len(samples)) + torch.logsumexp(log_likelihoods, dim=0)

# See arXiv:1502.05336 (also used in arXiv:1612.01474)
class CubicToyDataset(RegressionToyDataset):
    def __init__(self, min: float = -4, max: float = 4, sample_count: int = 20, normalize: bool = True, noise: float = 3, offset: float = 0, skip: float = 0):
        self.offset = offset
        super().__init__(min, max, sample_count, normalize, noise, skip)

    def eval(self, value, noise):
        return value**3 + self.offset + noise

class TrigonometricToyDataset(RegressionToyDataset):
    def __init__(self, data_areas = [(0, 0.5, 100)], noise = 0.02):
        super().__init__(data_areas, noise)

    def eval(self, value, noise):
        #return value + 0.3*torch.sin(2*torch.pi*(value + noise)) + 0.3*torch.sin(4*torch.pi*(value + noise)) + noise
        return value + 0.3*torch.sin(2*torch.pi*(value)) + 0.3*torch.sin(4*torch.pi*(value)) + noise

class ClassificationToyDataset:
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels

    def dataset(self):
        return torch.utils.data.TensorDataset(self.samples, self.labels)

class TwoMoonsDataset(ClassificationToyDataset):
    def __init__(self, samples: int = 100, noise: float = 0.1, seed: int = 42, extra_samples=0):
        data, labels = sklearn.datasets.make_moons(samples, noise=noise, random_state=seed)
        if extra_samples > 0:
            data = np.append(data, np.array([np.array([3 * np.cos(t) + 2, 3 * np.sin(t)]) for t in np.linspace(np.pi, 1.15 * np.pi, extra_samples)]), axis=0)
            labels = np.append(labels, np.array([np.array(0) for _ in range(extra_samples)]), axis=0)

            data = np.append(data, np.array([np.array([3 * np.cos(t) - 1, 3 * np.sin(t)]) for t in np.linspace(0, 0.15 * np.pi, extra_samples)]), axis=0)
            labels = np.append(labels, np.array([np.array(1) for _ in range(extra_samples)]), axis=0)

        super().__init__(torch.tensor(data, dtype=torch.float), torch.tensor(labels, dtype=torch.float).unsqueeze(-1))