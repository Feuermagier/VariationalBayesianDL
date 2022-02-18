import torch
import numpy as np

# See arXiv:1502.05336 (also used in arXiv:1612.01474)
class CubicToyDataset(torch.utils.data.IterableDataset):

    def __init__(self, min: float = -4, max: float = 4, sample_count: int = 20, rng = None):
        super().__init__()
        if rng is None:
            rng = np.random.default_rng()
        xs = rng.uniform(min, max, sample_count).astype(np.float32)
        self.samples = [(torch.tensor([x], dtype=torch.float), torch.tensor([x**3 + rng.normal(0, 9)], dtype=torch.float)) for x in xs]
    
    def __iter__(self):
        return iter(self.samples)
