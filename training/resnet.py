import torch
import torch.nn as nn
from training.dropout import FixableDropout
from training.bbb_layers import BBBConvolution, BBBLinear

class PreBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.main_path = nn.Sequential(
            nn.BatchNorm2d(in_channels, track_running_stats=False),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, track_running_stats=False),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        )

        nn.init.kaiming_normal_(self.main_path[2].weight.data)
        nn.init.kaiming_normal_(self.main_path[5].weight.data)

        if stride != 1:
            self.skip_path = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        else:
            self.skip_path = nn.Identity()

    def forward(self, input):
        return self.main_path(input) + self.skip_path(input)

class PreResNet(nn.Module):
    def __init__(self, in_size, in_channels, classes):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),

            PreBasicBlock(16, 16, 1),
            PreBasicBlock(16, 16, 1),
            PreBasicBlock(16, 16, 1),

            PreBasicBlock(16, 32, 2),
            PreBasicBlock(32, 32, 1),
            PreBasicBlock(32, 32, 1),

            PreBasicBlock(32, 64, 2),
            PreBasicBlock(64, 64, 1),
            PreBasicBlock(64, 64, 1),

            nn.BatchNorm2d(64, track_running_stats=False),
            nn.ReLU(),
            nn.AvgPool2d(8) if in_size >= 32 else nn.Identity(),

            nn.Flatten(),
            nn.Linear(64 * (in_size // (32 if in_size >= 32 else 4))**2, classes)
        )

    def forward(self, input):
        return self.model(input)

class DropoutPreBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, p, stride=1):
        super().__init__()

        self.main_path = nn.Sequential(
            nn.BatchNorm2d(in_channels, track_running_stats=False),
            nn.ReLU(),
            FixableDropout(p),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, track_running_stats=False),
            nn.ReLU(),
            FixableDropout(p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        )

        if stride != 1:
            self.skip_path = nn.Sequential(
                FixableDropout(p),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            )
        else:
            self.skip_path = nn.Identity()

    def forward(self, input):
        return self.main_path(input) + self.skip_path(input)

class DropoutPreResNet(nn.Module):
    def __init__(self, in_size, in_channels, classes, p):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),

            DropoutPreBasicBlock(16, 16, p, 1),
            DropoutPreBasicBlock(16, 16, p, 1),
            DropoutPreBasicBlock(16, 16, p, 1),

            DropoutPreBasicBlock(16, 32, p, 2),
            DropoutPreBasicBlock(32, 32, p, 1),
            DropoutPreBasicBlock(32, 32, p, 1),

            DropoutPreBasicBlock(32, 64, p, 2),
            DropoutPreBasicBlock(64, 64, p, 1),
            DropoutPreBasicBlock(64, 64, p, 1),

            nn.BatchNorm2d(64, track_running_stats=False),
            nn.ReLU(),
            nn.AvgPool2d(8) if in_size >= 32 else nn.Identity(),

            nn.Flatten(),
            FixableDropout(p),
            nn.Linear(64 * (in_size // (32 if in_size >= 32 else 4))**2, classes)
        )

    def forward(self, input):
        return self.model(input)

class VariationalPreBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, prior, stride=1):
        super().__init__()

        self.main_path = nn.Sequential(
            nn.BatchNorm2d(in_channels, track_running_stats=False),
            nn.ReLU(),
            BBBConvolution(in_channels, out_channels, kernel_size=3, stride=stride, bias=False, weight_prior=prior, bias_prior=prior, padding=1),
            nn.BatchNorm2d(out_channels, track_running_stats=False),
            nn.ReLU(),
            BBBConvolution(out_channels, out_channels, kernel_size=3, stride=1, bias=False, weight_prior=prior, bias_prior=prior, padding=1)
        )

        nn.init.kaiming_normal_(self.main_path[2].weight_mu.data)
        nn.init.kaiming_normal_(self.main_path[5].weight_mu.data)

        if stride != 1:
            self.skip_path = BBBConvolution(in_channels, out_channels, kernel_size=1, stride=stride, bias=False, weight_prior=prior, bias_prior=prior)
        else:
            self.skip_path = nn.Identity()

    def forward(self, input):
        return self.main_path(input) + self.skip_path(input)

class VariationalPreResNet(nn.Module):
    def __init__(self, in_size, in_channels, classes, prior):
        super().__init__()

        self.model = nn.Sequential(
            BBBConvolution(in_channels, 16, kernel_size=3, stride=1, weight_prior=prior, bias_prior=prior, padding=1),

            VariationalPreBasicBlock(16, 16, prior, 1),
            VariationalPreBasicBlock(16, 16, prior, 1),
            VariationalPreBasicBlock(16, 16, prior, 1),

            VariationalPreBasicBlock(16, 32, prior, 2),
            VariationalPreBasicBlock(32, 32, prior, 1),
            VariationalPreBasicBlock(32, 32, prior, 1),

            VariationalPreBasicBlock(32, 64, prior, 2),
            VariationalPreBasicBlock(64, 64, prior, 1),
            VariationalPreBasicBlock(64, 64, prior, 1),

            nn.BatchNorm2d(64, track_running_stats=False),
            nn.ReLU(),
            nn.AvgPool2d(8) if in_size >= 32 else nn.Identity(),

            nn.Flatten(),
            BBBLinear(64 * (in_size // (32 if in_size >= 32 else 4))**2, classes, prior, prior)
        )

    def forward(self, input):
        return self.model(input)