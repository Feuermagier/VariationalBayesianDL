import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
import matplotlib.pyplot as plt

from training import util
from training.calibration import reliability_diagram


def eval_model(name, eval_fn, losses, samples, testloader, device, include_ace=True):
    torch.manual_seed(42)
    # Test performance
    errors = []
    confidences = []
    with torch.no_grad():
        for data, target in testloader:
            output = eval_fn(data.to(device), samples).mean(dim=0).cpu()
            preds = torch.argmax(output, dim=1)
            errors.append(preds == target)
            confidences.append(output[torch.arange(output.shape[0]), preds].exp())
            # outputs = eval_fn(data.to(device), samples).cpu()
            # sample_preds = torch.transpose(
            #     torch.argmax(outputs, dim=2), 0, 1)
            # preds = torch.mode(sample_preds, dim=1)[0]
            # errors = torch.cat((errors, preds == target))
            # confs = outputs[:, torch.arange(
            #     outputs.shape[1]), preds].mean(dim=0).exp()
            # confidences = torch.cat((confidences, confs))
    errors = torch.cat(errors)
    confidences = torch.cat(confidences)
    accuracy = errors.sum() / len(errors)

    fig = plt.figure(figsize=(12, 5))
    fig.set_tight_layout(True)
    fig.suptitle(name + f" (Test Accuracy {accuracy:.3f})", fontsize=16)

    # Plot loss over time
    max_epochs = max([len(l) for l in losses])
    loss_ax = fig.add_subplot(1, 2, 1)
    loss_ax.set_xlabel("Epoch", fontsize=14)
    loss_ax.set_xticks(np.arange(1, max_epochs + 1, 1))
    loss_ax.set_ylabel("Training NLL Loss", fontsize=14)
    for single_losses in losses:
        loss_ax.plot(np.arange(1, max_epochs + 1, 1), single_losses)

    rel_ax = fig.add_subplot(1, 2, 2)
    reliability_diagram(10, errors, confidences, rel_ax, include_ace)

    return fig

# models = [(name, eval_fn, loss_over_time, [ece_over_time per dataset in the same order], eval_samples)]
# datasets = [(name, dataloader)]


def eval_multiple(models, datasets, device, include_ace=True, include_mce=False):
    torch.manual_seed(42)
    width = len(models)
    height = len(datasets) + 1  # 2 * len(dataset) + 1
    fig = plt.figure(figsize=(8 * width, 5 * height))
    #fig.suptitle(name + f" (Test Accuracy {accuracy:.3f})", fontsize=16)

    for i, (name, _, loss_over_time, _, _) in enumerate(models):
        loss_ax = fig.add_subplot(height, width, i + 1)
        loss_ax.annotate(name, xy=(0.5, 1), xytext=(0, 10), xycoords="axes fraction",
                         textcoords="offset points",  ha="center", va="center", fontsize=16)
        util.plot_losses(name, loss_over_time, loss_ax)

    for i, (name, loader) in enumerate(datasets):
        for j, (_, eval_fn, _, eces, eval_samples) in enumerate(models):
            # ece_ax = fig.add_subplot(height, width, (2 * i + 1) * width + j + 1)
            # ece_ax.set_xlabel("Epoch", fontsize=14)
            # ece_ax.set_xticks(np.arange(1, len(eces) + 1, 1))
            # ece_ax.set_ylabel("ECE", fontsize=14)
            # if eces != []:
            #     ece_ax.plot(np.arange(1, len(eces) + 1, 1), eces)

            rel_ax = fig.add_subplot(height, width, (i + 1) * width + j + 1)
            errors = []
            confidences = []
            with torch.no_grad():
                for data, target in loader:
                    output = eval_fn(data.to(device), eval_samples).mean(dim=0).cpu()
                    preds = torch.argmax(output, dim=1)
                    errors.append(preds == target)
                    confidences.append(output[torch.arange(output.shape[0]), preds].exp())
                    # outputs = eval_fn(data.to(device), eval_samples).cpu()
                    # sample_preds = torch.transpose(
                    #     torch.argmax(outputs, dim=2), 0, 1)
                    # preds = torch.mode(sample_preds, dim=1)[0]
                    # errors = torch.cat((errors, preds == target))
                    # confs = outputs[:, torch.arange(
                    #     outputs.shape[1]), preds].mean(dim=0).exp()
                    # confidences = torch.cat((confidences, confs))
            errors = torch.cat(errors)
            confidences = torch.cat(confidences)
            reliability_diagram(10, errors, confidences, rel_ax, include_ace, include_mce)

            if j == 0:
                rel_ax.annotate(name, xy=(0, 0.5), xytext=(-rel_ax.yaxis.labelpad - 10, 0),
                                xycoords=rel_ax.yaxis.label, textcoords="offset points", fontsize=16, ha="left", va="center")

    fig.tight_layout()
    fig.subplots_adjust(left=0.2, top=0.95)
    return fig