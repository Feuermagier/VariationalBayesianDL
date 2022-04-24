import torch
import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch.nn.functional as F
from scipy.stats import wasserstein_distance
import math
from .util import gauss_logprob

class RegressionToyDataset(torch.utils.data.Dataset):
    def __init__(self, min: float, max: float, sample_count: int, normalize: bool, noise: float, skip):
        super().__init__()
        self.min, self.max = min, max

        if normalize:
            self.x_norm = 1 / np.abs(max)
            self.y_norm = 1 / np.abs(self.eval_value(torch.tensor(max)))
        else:
            self.x_norm = 1
            self.y_norm = 1

        self.xs, self.ys = _sample_from_fn(self.eval, min, max, sample_count, noise, skip)
        self.normalized_xs, self.normalized_ys = self.xs * self.x_norm, self.ys * self.y_norm
        xs, ys = torch.unsqueeze(self.normalized_xs, -1), torch.unsqueeze(self.normalized_ys, -1)
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

    def generate_eval_range(self, extra_range):
        extra = (self.max - self.min) * extra_range
        min, max = self.min - extra, self.max + extra
        return torch.linspace(min, max, 100)

    def generate_samples(self, eval_fn, samples, t):
        with torch.no_grad():
            outputs = eval_fn(torch.unsqueeze(t * self.x_norm, -1), samples)
        return outputs

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



    def plot_dataset(self, axis, extra_range=0.01):
        extra = (self.max - self.min) * extra_range
        min, max = self.min - extra, self.max + extra
        #plt.xlim(min, max)
        t = torch.linspace(min, max, 50)
        y = self.eval_value(t)
        axis.plot(t, y, color="blue")
        axis.scatter(self.xs, self.ys, s=4, color="blue")

def calculate_lml_gaussian(target, samples, var):
    assert len(samples) > 0
    log_likelihoods = torch.empty(len(samples))
    for i, mean in enumerate(samples):
        log_likelihoods[i] = gauss_logprob(mean, var, target).sum()
    return -math.log(len(samples)) + torch.logsumexp(log_likelihoods, dim=0)

def _sample_from_fn(function, min, max, sample_count, noise_sigma, skip=0):
    lower_xs = ((min + max - skip) / 2 - min) * torch.rand(math.floor(sample_count / 2)) + min
    higher_xs = (max - (min + max + skip) / 2) * torch.rand(math.ceil(sample_count / 2)) + (min + max + skip) / 2
    xs = torch.cat((lower_xs, higher_xs))
    noise = torch.normal(mean=torch.zeros(sample_count), std=torch.full((sample_count,), noise_sigma))
    ys = function(xs, noise)
    return xs.float(), ys.float()

# See arXiv:1502.05336 (also used in arXiv:1612.01474)
class CubicToyDataset(RegressionToyDataset):
    def __init__(self, min: float = -4, max: float = 4, sample_count: int = 20, normalize: bool = True, noise: float = 3, offset: float = 0, skip: float = 0):
        self.offset = offset
        super().__init__(min, max, sample_count, normalize, noise, skip)

    def eval(self, value, noise):
        return value**3 + self.offset + noise

class TrigonometricToyDataset(RegressionToyDataset):
    def __init__(self, min: float = 0, max: float = 0.5, sample_count: int = 20, normalize: bool = False, noise: float = 0.02, skip: float = 0):
        super().__init__(min, max, sample_count, normalize, noise, skip)

    def eval(self, value, noise):
        return value + 0.3*torch.sin(2*torch.pi*(value + noise)) + 0.3*torch.sin(4*torch.pi*(value + noise)) + noise

eval_points = 100
sample_cmap = ListedColormap(["red", "blue"])
area_cmap = plt.cm.RdBu
variance_cmap = plt.cm.viridis

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

    def plot_dataset(self, ax):
        data, labels = zip(*self)
        ax.scatter(*zip(*data), c=labels, cmap=sample_cmap, edgecolors="black")

    def plot(self, name, eval_fn, samples, xlim, ylim):
        with torch.no_grad():
            fig, ((value_ax, var_ax), (rel_ax, _)) = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(name)
            value_ax.set_xlim(-xlim, xlim)
            value_ax.set_ylim(-ylim, ylim)
            var_ax.set_xlim(-xlim, xlim)
            var_ax.set_ylim(-ylim, ylim)
            
            # Grid evaluation
            xs, ys = np.meshgrid(np.linspace(-xlim, xlim, eval_points), np.linspace(-ylim, ylim, eval_points))
            data = np.dstack((xs.reshape(eval_points * eval_points), ys.reshape(eval_points * eval_points)))[0]
            results = torch.stack(eval_fn(torch.from_numpy(data).float(), samples)).reshape((samples, eval_points, eval_points))
            value_ax.contourf(xs, ys, results.mean(dim=0), 100, cmap=area_cmap)
            var_ax.contourf(xs, ys, results.var(dim=0), 100, cmap=variance_cmap, vmin=0, vmax=1.0)

            # Training samples
            results = torch.stack(eval_fn(self.samples, samples))
            predictions = torch.round(torch.round(results).mean(dim=0))
            confidences = (2 * torch.abs(results - 0.5)).mean(dim=0)
            errors = predictions == self.labels
            value_ax.scatter(*zip(*self.samples), facecolors=sample_cmap(predictions), edgecolors=sample_cmap(self.labels))

            # Reliability diagram
            bin_count = 10
            bins = [[] for _ in range(bin_count)]
            for i, confidence in enumerate(confidences):
                bins[torch.floor(confidence * bin_count).int()].append(i)
            bin_accuracys = np.array([errors[bin].sum() / len(bin) if len(bin) > 0 else 0 for bin in bins])
            mid = np.linspace(0, 1 - 1 / bin_count, bin_count)
            bin_errors = np.abs(np.array(bin_accuracys) - mid)
            bin_confidences = np.array([confidences[bin].sum() / len(bin) if len(bin) > 0 else 0 for bin in bins])

            rel_ax.set_xlim(0, 1)
            rel_ax.set_ylim(0, 1)
            rel_ax.grid(color="tab:grey", linestyle=(0, (1, 5)), linewidth=1)
            interval = np.arange(0, 1, 1 / bin_count)
            rel_ax.bar(interval, bin_accuracys, 1 / bin_count, align="edge", color="b", edgecolor="k")
            rel_ax.bar(interval, bin_errors, 1 / bin_count, bottom=np.minimum(bin_accuracys, mid), align="edge", color="mistyrose", alpha=0.5, edgecolor="r", hatch="/")
            rel_ax.set_ylabel('Accuracy',fontsize=16)
            rel_ax.set_xlabel('Confidence',fontsize=16)

            ece = np.mean(np.abs(bin_accuracys - bin_confidences))
            mce = np.max(np.abs(bin_accuracys - bin_confidences))

            ident = [0.0, 1.0]
            rel_ax.plot(ident,ident,linestyle='--',color="tab:grey")

            text = f"{samples} weight sample(s)\n" \
                + f"Accuracy (majority vote): {errors.sum() / len(self.samples)} \n" \
                + f"ECE: {ece} \n" \
                + f"MCE: {mce}"
            fig.text(0.5, 0.01, text, ha="center", va="top", fontsize=12)

class TwoMoonsDataset(ClassificationToyDataset):
    def __init__(self, samples: int = 100, noise: float = 0.1, seed: int = None, extra_samples=0):
        #all_data, all_labels = np.zeros((0, 2)), np.zeros((0))
        #for x in range(2):
        #    for y in range(2):
        #        data, labels = sklearn.datasets.make_moons(samples, noise=noise, random_state=seed)
        #        all_data = np.append(all_data, data + np.array([x * 4 - 2, y * 4 - 2]), axis=0)
        #        all_labels = np.append(all_labels, labels, axis=0)
        #super().__init__(torch.tensor(all_data, dtype=torch.float), torch.tensor(all_labels, dtype=torch.float).unsqueeze(-1))

        data, labels = sklearn.datasets.make_moons(samples, noise=noise, random_state=seed)
        if extra_samples > 0:
            data = np.append(data, np.array([np.array([3 * np.cos(t) + 2, 3 * np.sin(t)]) for t in np.linspace(np.pi, 1.15 * np.pi, extra_samples)]), axis=0)
            labels = np.append(labels, np.array([np.array(0) for _ in range(extra_samples)]), axis=0)

            data = np.append(data, np.array([np.array([3 * np.cos(t) - 1, 3 * np.sin(t)]) for t in np.linspace(0, 0.15 * np.pi, extra_samples)]), axis=0)
            labels = np.append(labels, np.array([np.array(1) for _ in range(extra_samples)]), axis=0)

        super().__init__(torch.tensor(data, dtype=torch.float), torch.tensor(labels, dtype=torch.float).unsqueeze(-1))