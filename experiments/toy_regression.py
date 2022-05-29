import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from training import util
from training import toy
from training.regresssion import RegressionResults, plot_calibration

NOISE = torch.tensor(0.02)

def wrap(model):
    return util.GaussWrapper(model, NOISE, False)

def plot_grid(dataset, testset, models, device, min=-0.3, max=0.8):
    fig, axes = plt.subplots(nrows=4, ncols=len(models), figsize=(5 * len(models), 15), squeeze=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=testset.tensors[0].shape[0])
    results = []
    for i, (name, model, samples) in enumerate(models):
        util.plot_losses(name, model.all_losses(), axes[0, i])
        result = RegressionResults(testloader, name, model.infer, samples, device, fit_gaussian=False, target_mean=dataset.y_mean, target_std = dataset.y_std)
        dataset.plot_predictions(min, max, model.infer, samples, axes[1, i], dataset=None, alpha = 0.1 if samples > 10 else 1)
        dataset.plot_predictions(min, max, model.infer, samples, axes[2, i], dataset=testset, alpha = 0.1 if samples > 10 else 1)
        plot_calibration(name, result, axes[3, i])
        results.append(result)
        print(f"Test LML ({name}): {result.average_lml}")

    return fig, results

def gap_datasets(device):
    dataset = toy.TrigonometricToyDataset([(0, 0.13, 100), (0.4, 0.55, 100)], NOISE)
    trainloader = torch.utils.data.DataLoader(dataset.trainset, batch_size=20)
    testset = dataset.generate_testset(0.0, 0.55, 500, NOISE)
    #testloader = torch.utils.data.DataLoader(testset, batch_size=testset.tensors[0].shape[0])

    return dataset, trainloader, testset

def store_results(dataset, testset, models, device, min=-0.3, max=0.8):
    testloader = torch.utils.data.DataLoader(testset, batch_size=testset.tensors[0].shape[0])
    for i, (name, model, samples) in enumerate(models):
        result = RegressionResults(testloader, name, model.infer, samples, device, fit_gaussian=False, target_mean=dataset.y_mean, target_std = dataset.y_std)
        fig, ax = plt.subplots(1, 1)
        dataset.plot_predictions(min, max, model.infer, samples, ax, dataset=None, alpha = 0.1 if samples > 10 else 1)
        fig.tight_layout(pad=0)
        fig.savefig(f"results/toy/regression/{name}_plot.pdf")

        fig, ax = plt.subplots(1, 1)
        plot_calibration(None, result, ax, include_text=False)
        fig.tight_layout(pad=0)
        fig.savefig(f"results/toy/regression/{name}_calibration.pdf")