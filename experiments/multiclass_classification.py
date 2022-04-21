from dataclasses import dataclass
from itertools import repeat
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import importlib
import time
import math
import matplotlib.pyplot as plt
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, get_kl_loss

from training import util
from training.swag import SWAGWrapper
from training.bbb import BBBConvolution, run_bbb_epoch, BBBLinear, GaussianPrior
from training.calibration import reliability_diagram

def eval_model(name, eval_fn, losses, samples, testloader, device, include_ace=True):
    # Test performance
    errors = torch.empty(0)
    confidences = torch.empty(0)
    with torch.no_grad():
        for data, target in testloader:
            outputs = torch.stack(eval_fn(data.to(device), samples)).cpu()
            sample_preds = torch.transpose(torch.argmax(outputs, dim=2), 0, 1)
            preds = torch.mode(sample_preds, dim=1)[0]
            errors = torch.cat((errors, preds == target))
            confs = outputs[:,torch.arange(outputs.shape[1]),preds].mean(dim=0).exp()
            confidences = torch.cat((confidences, confs))
    accuracy = errors.sum() / len(errors)

    fig = plt.figure(figsize=(12, 5))
    fig.set_tight_layout(True)
    fig.suptitle(name + f" (Test Accuracy {accuracy:.3f})", fontsize=16)
    epochs = len(losses)

    # Plot loss over time
    loss_ax = fig.add_subplot(1, 2, 1)
    loss_ax.set_xlabel("Epoch", fontsize=14)
    loss_ax.set_xticks(np.arange(1, epochs + 1, 1))
    loss_ax.set_ylabel("Training NLL Loss", fontsize=14)
    loss_ax.plot(np.arange(1, epochs + 1, 1), losses)

    rel_ax = fig.add_subplot(1, 2, 2)
    reliability_diagram(10, errors, confidences, rel_ax, include_ace)

    return fig

# models = [(name, eval_fn, loss_over_time, [ece_over_time per dataset in the same order], eval_samples)]
# datasets = [(name, dataloader)]
def eval_multiple(models, datasets, device, include_ace=True, include_mce=False):
    width = len(models)
    height = len(datasets) + 1 # 2 * len(dataset) + 1
    fig = plt.figure(figsize=(8 * width, 5 * height))
    #fig.suptitle(name + f" (Test Accuracy {accuracy:.3f})", fontsize=16)

    for i, (name, _, loss_over_time, _, _) in enumerate(models):
        loss_ax = fig.add_subplot(height, width, i + 1)
        loss_ax.annotate(name, xy=(0.5, 1), xytext=(0, 10), xycoords="axes fraction", textcoords="offset points",  ha="center", va="center", fontsize=16)
        loss_ax.set_xlabel("Epoch", fontsize=14)
        loss_ax.set_xticks(np.arange(1, len(loss_over_time) + 1, 1))
        loss_ax.set_ylabel("Training NLL Loss", fontsize=14)
        loss_ax.plot(np.arange(1, len(loss_over_time) + 1, 1), loss_over_time)

    for i, (name, loader) in enumerate(datasets):
        for j, (_, eval_fn, _, eces, eval_samples) in enumerate(models):
            # ece_ax = fig.add_subplot(height, width, (2 * i + 1) * width + j + 1)
            # ece_ax.set_xlabel("Epoch", fontsize=14)
            # ece_ax.set_xticks(np.arange(1, len(eces) + 1, 1))
            # ece_ax.set_ylabel("ECE", fontsize=14)
            # if eces != []:
            #     ece_ax.plot(np.arange(1, len(eces) + 1, 1), eces)

            rel_ax = fig.add_subplot(height, width, (i + 1) * width + j + 1)
            errors = torch.empty(0)
            confidences = torch.empty(0)
            with torch.no_grad():
                for data, target in loader:
                    outputs = torch.stack(eval_fn(data.to(device), eval_samples)).cpu()
                    sample_preds = torch.transpose(torch.argmax(outputs, dim=2), 0, 1)
                    preds = torch.mode(sample_preds, dim=1)[0]
                    errors = torch.cat((errors, preds == target))
                    confs = outputs[:,torch.arange(outputs.shape[1]),preds].mean(dim=0).exp()
                    confidences = torch.cat((confidences, confs))
            reliability_diagram(10, errors, confidences, rel_ax)

            if j == 0:
                rel_ax.annotate(name, xy=(0, 0.5), xytext=(-rel_ax.yaxis.labelpad - 10, 0), xycoords=rel_ax.yaxis.label, textcoords="offset points", fontsize=16, ha="left", va="center")

    fig.tight_layout()
    fig.subplots_adjust(left=0.2, top=0.95)
    return fig

def point_predictor(layers, epochs, dataloader, batch_size, device):
    pp_model = util.generate_model(layers)
    pp_model.to(device)

    optimizer = torch.optim.SGD(pp_model.parameters(), lr=0.01)
    losses = []
    for epoch in range(epochs):
        epoch_loss = torch.tensor(0, dtype=torch.float)
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = pp_model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.cpu()
        epoch_loss /= (len(dataloader) * batch_size)
        losses.append(epoch_loss.detach())
        if epoch % 1 == 0:
            print(f"Epoch {epoch}: NLL loss {epoch_loss}")
    print(f"Final loss {epoch_loss}")

    def eval_pp(input, samples):
        return [pp_model(input) for _ in range(samples)]

    return eval_pp, losses

def swag(layers, epochs, dataloader, batch_size, swag_config, device, use_lr_cycles = False):
    swag_model = util.generate_model(layers)
    swag_model.to(device)
    optimizer = torch.optim.SGD(swag_model.parameters(), lr=0.01)
    wrapper = SWAGWrapper(swag_model, optimizer, swag_config, device, use_lr_cycles)
    losses = []
    for epoch in range(epochs):
        epoch_loss = torch.tensor(0, dtype=torch.float)
        for i, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = swag_model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.cpu()
            wrapper.update(epoch, i)
        epoch_loss /= (len(dataloader) * batch_size)
        losses.append(epoch_loss.detach())
        if epoch % 1 == 0:
            print(f"Epoch {epoch}: NLL loss {epoch_loss}")
    print(f"Final loss {epoch_loss}")
    wrapper.report_status()

    def eval_swag(input, samples):
        torch.manual_seed(42)
        return [wrapper.sample(input) for _ in range(samples)]
    
    return eval_swag, losses

def ensemble(ensemble_count, layers, epochs, dataloader, batch_size):
    models = [util.generate_model(layers, "relu", "sigmoid") for _ in range(ensemble_count)]

    for i, model in enumerate(models):
        print(f"Training model {i}")
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        for epoch in range(epochs):
            epoch_loss = torch.tensor(0, dtype=torch.float)
            for data, target in dataloader:
                optimizer.zero_grad()
                output = model(data)
                loss = F.binary_cross_entropy(output, target)
                loss.backward()
                optimizer.step()
                epoch_loss += loss
            if epoch % 100 == 0:
                print(f"  Epoch {epoch}: loss {epoch_loss / (len(dataloader) * batch_size)}")
        print(f"Final loss {epoch_loss / (len(dataloader) * batch_size)}")


    def eval_esemble(input, samples):
        assert samples == len(models)
        return [model(input) for model in models]

    return eval_esemble

def mc_droupout(p, layers, epochs, dataloader, batch_size):
    mc_model = util.generate_model(layers, "relu", "sigmoid", scale=1/(1 - p), dropout_p=p)
    optimizer = torch.optim.SGD(mc_model.parameters(), lr=0.01)

    for epoch in range(epochs):
        mc_model.train()
        epoch_loss = torch.tensor(0, dtype=torch.float)
        for data, target in dataloader:
            optimizer.zero_grad()
            output = mc_model(data)
            loss = F.binary_cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: loss {epoch_loss / (len(dataloader) * batch_size)}")
    print(f"Final loss {epoch_loss / (len(dataloader) * batch_size)}")

    def eval_dropout(input, samples):
        mc_model.train() # Enable dropout
        return [mc_model(input) for _ in range(samples)]

    return eval_dropout

def bbb(prior, sampling, mc_samples, kl_rescaling, layers, epochs, dataloader, batch_size, device):
    linear_fn = lambda i, o: BBBLinear(i, o, prior, prior, device, sampling=sampling)
    conv_fn = lambda i, o, k: BBBConvolution(i, o, k, prior, prior, device, sampling=sampling)
    bbb_model = util.generate_model(layers, linear_fn=linear_fn, conv_fn=conv_fn)
    bbb_model.to(device)
    optimizer = torch.optim.SGD(bbb_model.parameters(), lr=0.001)
    loss_fn = torch.nn.NLLLoss(reduction="sum")

    losses = []
    for epoch in range(epochs):
        loss = run_bbb_epoch(bbb_model, optimizer, loss_fn, dataloader, device, samples=mc_samples, kl_rescaling=kl_rescaling)
        loss /= len(dataloader) * batch_size
        losses.append(loss.detach())
        if epoch % 1 == 0:
            print(f"Epoch {epoch}: loss {loss}")
    print(f"Final loss {loss}")


    def eval_bbb(input, samples):
        return [bbb_model(input) for _ in range(samples)]
    
    return eval_bbb, losses

def bbb_intel(config, layers, epochs, dataloader, batch_size):
    intel_model = util.generate_model(layers, "relu", "sigmoid")

    dnn_to_bnn(intel_model, config)

    optimizer = torch.optim.Adam(intel_model.parameters(), 0.01)
    for epoch in range(epochs):
        intel_model.train()
        epoch_loss = torch.tensor(0, dtype=torch.float)
        for data, target in dataloader:
            optimizer.zero_grad()
            output = intel_model(data)
            loss = get_kl_loss(intel_model) / batch_size + F.binary_cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: loss {epoch_loss / (len(dataloader) * batch_size)}")
    print(f"Final loss {epoch_loss / (len(dataloader) * batch_size)}")

    def eval_bayes(input, samples):
        intel_model.eval()
        return [intel_model(input) for _ in range(samples)]

    return eval_bayes