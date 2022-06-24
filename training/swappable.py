import torch
import torch.nn as nn
import torch.nn.functional as F


class SwappableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        