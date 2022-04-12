import multiprocessing
from itertools import repeat
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms
import itertools
import importlib
import time
import math
from training import util
from training.swag import SWAGWrapper
from training.bbb import run_bbb_epoch, BBBLinear, GaussianPrior
import gpytorch
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, get_kl_loss

def gaussian_process(epochs, xs, ys):
    class GPModel(gpytorch.models.ExactGP):
        def __init__(self, likelihood):
            super().__init__(xs, ys, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    gp = GPModel(likelihood)

    gp.train()
    likelihood.train()
    optimizer = torch.optim.Adam(gp.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp)

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = gp(xs)
        loss = -mll(output, ys)
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: loss {loss}")

    def eval_gp(input, samples):
        gp.eval()
        likelihood.eval()
        with torch.no_grad():
            dist = gp(input.squeeze(-1))
            outputs = [(dist.sample(), likelihood.noise) for _ in range(samples)]
            return outputs

    def gp_true_lml(input, y):
        with torch.no_grad():
            dist = gp(input.squeeze(-1))
            return mll(dist, y) * len(input)

    return eval_gp, gp_true_lml

def point_estimator(layers, noise, learn_var, epochs, dataloader, batch_size, device):
    pp_model = util.GaussWrapper(util.generate_model(layers, "relu", None), noise, learn_var)
    pp_model.to(device)
    optimizer = torch.optim.SGD(pp_model.parameters(), lr=0.01)
    for epoch in range(epochs):
        epoch_loss = torch.tensor(0, dtype=torch.float)
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            mean, var = pp_model(data)
            loss = F.gaussian_nll_loss(mean, target, var)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.cpu()
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: loss {epoch_loss / (len(dataloader) * batch_size)}")
    print(f"Final loss {epoch_loss / (len(dataloader) * batch_size)}")

    def eval_pp(input, samples):
        return [pp_model(input) for _ in range(samples)]

    return eval_pp

def swag(layers, noise, learn_var, epochs, swag_config, dataloader, batch_size, device):
    swag_model = util.GaussWrapper(util.generate_model(layers, "relu", None), noise, learn_var)
    optimizer = torch.optim.SGD(swag_model.parameters(), lr=0.01)
    wrapper = SWAGWrapper(swag_model, swag_config, device)

    for epoch in range(epochs):
        epoch_loss = torch.tensor(0, dtype=torch.float)
        for i, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            mean, var = swag_model(data)
            loss = F.gaussian_nll_loss(mean, target, var)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.cpu()
            wrapper.update(epoch, i)
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: loss {epoch_loss / (len(dataloader) * batch_size)}")
    print(f"Final loss {epoch_loss / (len(dataloader) * batch_size)}")

    def eval_swag(input, samples):
        return [wrapper.sample(swag_model, input) for _ in range(samples)]

    return eval_swag

def _train_ensemble_member(ind, common):
    i, model = ind
    epochs, dataloader, batch_size = common
    print(f"Training model {i}")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for epoch in range(epochs):
        epoch_loss = torch.tensor(0, dtype=torch.float)
        for data, target in dataloader:
            optimizer.zero_grad()
            mean, var = model(data)
            loss = F.gaussian_nll_loss(mean, target, var)
            loss.backward()
            optimizer.step()
            epoch_loss += loss
        if epoch % 100 == 0:
            print(f"  Epoch {epoch}: loss {epoch_loss / (len(dataloader) * batch_size)}")
    print(f"  Final loss {epoch_loss / (len(dataloader) * batch_size)}")
    return (epoch_loss / (len(dataloader) * batch_size)).detach()

def ensemble(ensemble_count, layers, noise, learn_var, epochs, dataloader, batch_size, parallel=True):
    ensemble_model = [util.GaussWrapper(util.generate_model(layers, "relu", None), noise, learn_var) for _ in range(ensemble_count)]

    if parallel:
        workers = min(ensemble_count, multiprocessing.cpu_count())
        print(f"Setting up {workers} workers")
        with multiprocessing.Pool(workers) as pool:
            print("Training in parallel")
            for loss in pool.starmap(_train_ensemble_member, zip(enumerate(ensemble_model), repeat((epochs, dataloader, batch_size)))):
                print(f"Final loss {loss}")
    else:
        for i, model in enumerate(ensemble_model):
            _train_ensemble_member((i, model), (epochs, dataloader, batch_size))

    def eval_esemble(input, samples):
        if samples % ensemble_count != 0:
            raise ValueError("samples must divisible by the number of ensembles")
        return [model(input) for model in ensemble_model] * (samples // ensemble_count)

    return eval_esemble

def mc_dropout(p, layers, noise, learn_var, epochs, dataloader, batch_size):
    dropout_model = util.GaussWrapper(util.generate_model(layers, "relu", None, scale=1/(1 - p), dropout_p=p), noise, learn_var)

    optimizer = torch.optim.SGD(dropout_model.parameters(), lr=0.01)
    for epoch in range(epochs):
        dropout_model.train()
        epoch_loss = torch.tensor(0, dtype=torch.float)
        for data, target in dataloader:
            optimizer.zero_grad()
            mean, var = dropout_model(data)
            loss = F.gaussian_nll_loss(mean, target, var)
            loss.backward()
            optimizer.step()
            epoch_loss += loss
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: loss {epoch_loss / (len(dataloader) * batch_size)}")
    print(f"Final loss {epoch_loss / (len(dataloader) * batch_size)}")

    def eval_dropout(input, samples):
        dropout_model.train() # Enable dropout
        return [dropout_model(input) for _ in range(samples)]

    return eval_dropout

def intel_bbb(layers, noise, learn_var, epochs, dataloader, batch_size):
    intel_model = util.GaussWrapper(util.generate_model(layers, "relu", None), noise, learn_var)
    dnn_to_bnn(intel_model, {
            "prior_mu": 0.0,
            "prior_sigma": 1.0,
            "posterior_mu_init": 0.0,
            "posterior_rho_init": -3.0,
            "type": "Reparameterization",
            "moped_enable": False,
            "moped_delta": 0.5,
    })

    optimizer = torch.optim.Adam(intel_model.parameters(), 0.01)
    for epoch in range(epochs):
        intel_model.train()
        epoch_loss = torch.tensor(0, dtype=torch.float)
        for data, target in dataloader:
            optimizer.zero_grad()
            mean, var = intel_model(data)
            loss = get_kl_loss(intel_model) / len(dataloader) + F.gaussian_nll_loss(mean, target, var)
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

def bbb(layers, noise, learn_var, epochs, dataloader, batch_size, device, sampling="activations", layer_samples=1, global_samples=5, prior=GaussianPrior(0, 1)):
    pi = 0.25  # 0.25, 0.5, 0.75
    sigma1 = np.exp(-0)  # 0, 1, 2
    sigma2 = np.exp(-6)  # 6, 7, 8
    #prior = util.GaussianMixture(pi, sigma1, sigma2)

    bbb_model = util.GaussWrapper(util.generate_model(layers, "relu", None, 
        linear_fn=lambda i, o: BBBLinear(i, o, prior, prior, device, mc_sample=layer_samples, sampling=sampling)), 
        noise, learn_var)
    bbb_model.to(device)
    optimizer = torch.optim.SGD(bbb_model.parameters(), lr=0.001, momentum=0.95)
    def uncurried_nll_loss(output, target):
        mean, var = output
        return F.gaussian_nll_loss(mean, target, var)

    for epoch in range(epochs):
        loss = run_bbb_epoch(bbb_model, optimizer, uncurried_nll_loss, dataloader, device, samples=global_samples)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: loss {loss / (len(dataloader) * batch_size)}")
    print(f"Final loss {loss / (len(dataloader) * batch_size)}")


    def eval_bbb(input, samples):
        bbb_model.eval()
        return [bbb_model(input) for _ in range(samples)]

    return eval_bbb