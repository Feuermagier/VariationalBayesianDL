import torch
import numpy as np

# See arXiv:1502.05336 (also used in arXiv:1612.01474)
class CubicToyDataset(torch.utils.data.Dataset):
    def __init__(self, min: float = -4, max: float = 4, sample_count: int = 20, normalize: bool = True, noise: float = 3, rng=None):
        super().__init__()
        if rng is None:
            rng = np.random.default_rng()

        if normalize:
            self.x_norm = 1 / np.abs(max)
            self.y_norm = 1 / np.abs(max ** 3)
        else:
            self.x_norm = 1
            self.y_norm = 1

        xs, ys = _sample_from_fn(
            rng, lambda x: x**3, min, max, sample_count, noise)
        xs, ys = torch.unsqueeze(xs, -1) * self.x_norm, torch.unsqueeze(ys, -1) * self.y_norm
        self.samples = [(x, y) for x, y in zip(xs, ys)] # Just using zip() doesn't work for some reason

    def __iter__(self):
        return iter(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, key):
        return self.samples[key]

    def eval_value(self, value):
        return value**3

def _sample_from_fn(rng: np.random.Generator, function, min, max, sample_count, noise_sigma):
    xs = rng.uniform(min, max, sample_count)
    ys = function(xs) + rng.normal(loc=0, scale=noise_sigma, size=sample_count)
    return torch.from_numpy(xs).float(), torch.from_numpy(ys).float()