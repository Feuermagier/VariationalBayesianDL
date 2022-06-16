import torch
import torch.nn as nn
import numpy as np
from torch.optim.optimizer import Optimizer, required
from functorch import vmap, grad_and_value, make_functional_with_buffers

from training.network import generate_model


############################# VOGN #################################
def vogn_init(parameters, N, init):
    states = []
    for param in parameters:
        state = {}
        state["lr"] = init["lr"]
        state["betas"] = init["betas"] if "betas" in init else (0.9, 0.999)
        state["prior_prec"] = init["prior_prec"] if "prior_prec" in init else 1
        state["damping"] = init["damping"] if "damping" in init else 0
        state["tempering"] = init["tempering"] if "tempering" in init else 1
        state["augmentation"] = init["augmentation"] if "augmentation" in init else 1
        state["bias_correction"] = init["bias_correction"] if "bias_correction" in init else False
        state["sample"] = init["sample"] if "sample" in init else True # Set this to False to use OGN

        state["N"] = N
        state["momentum"] = torch.zeros_like(param, device=param.device)
        state["scale"] = None
        state["step"] = 0

        states.append(state)
    return states

def vogn_prepare(parameters, states):
    perturbed_params = []
    for param, state in zip(parameters, states):
        if state["scale"] is not None and state["sample"] is True:
            delta = state["augmentation"] * state["prior_prec"] / state["N"]
            std = (1 / (delta + state["scale"] + state["damping"]) / state["N"]).sqrt()
            epsilon = torch.randn_like(param, device=param.device)
            perturbed_params.append(param + epsilon * std)
        else:
            perturbed_params.append(param)
    return perturbed_params

def vogn_step(parameters, grads, states):
    new_parameters = []
    with torch.no_grad():
        for grad, param, state in zip(grads, parameters, states):
            state["step"] += 1
            t = state["step"]
            beta1, beta2 = state["betas"]
            delta = state["augmentation"] * state["prior_prec"] / state["N"]

            grad = grad.mean(dim=0)
            avg_grad = grad.mean(dim=0)
            sq_grad = (grad**2).mean(dim=0)

            if state["scale"] is None:
                state["scale"] = sq_grad + delta + state["damping"] # We treat the first batch as our initialization batch
                new_parameters.append(param)
            else:
                state["momentum"] = (1 - beta1) * state["momentum"] + (avg_grad + delta * param)
                state["scale"] = (1 - state["tempering"] * beta2) * state["scale"] + beta2 * sq_grad
                update = state["lr"] * state["momentum"] / (state["scale"] + state["damping"] + delta)
                if state["bias_correction"]:
                    update *= (1 - beta2**t) / (1 - beta1**t)
                new_parameters.append(param - update)
    return new_parameters, states

def vogn_stds(states):
    stds = []
    for state in states:
        delta = state["augmentation"] * state["prior_prec"] / state["N"]
        stds.append((1 / (delta + state["scale"] + state["damping"]) / state["N"]).sqrt())
    return stds


class VOGNModule(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.model, self.params, self.buffs = make_functional_with_buffers(generate_model(layers))
        self.optim_state = None
        self.losses = []

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return {
            "params": self.params,
            "buffers": self.buffs,
            "optim_state": self.optim_state,
            "losses": self.losses
        }

    def load_state_dict(self, dict):
        self.params = dict["params"]
        self.buffs = dict["buffers"]
        self.optim_state = dict["optim_state"]
        self.losses = dict["losses"]

    def train_model(self, epochs, loss_fn, optim_params, loader, batch_size, device, report_every_epochs=1, mc_samples=10):
        self.params = [p.to(device) for p in self.params]
        self.buffs = [b.to(device) for b in self.buffs]

        self.optim_state = vogn_init(self.params, len(loader) * batch_size, optim_params)

        def get_loss(parameters, input, target):
            input = input.unsqueeze(0)
            target = target.unsqueeze(0)
            output = self.model(parameters, self.buffs, input)
            loss = loss_fn(output, target)
            return loss

        def run_sample(parameters, optim_state, input, target):
            prep_params = vogn_prepare(parameters, optim_state)
            return vmap(grad_and_value(get_loss), (None, 0, 0))(prep_params, input, target)

        for epoch in range(epochs):
            epoch_loss = torch.tensor(0, dtype=torch.float)
            for data, target in loader:
                data, target = data.to(device), target.to(device)
                data, target = data.expand(mc_samples, *data.shape), target.expand(mc_samples, *target.shape)

                grads, loss = vmap(run_sample, (None, None, 0, 0), randomness="different")(self.params, self.optim_state, data, target)
                loss = loss.mean()
                self.params, self.optim_state = vogn_step(self.params, grads, self.optim_state)
                
                epoch_loss += loss.cpu().item()
            epoch_loss /= len(loader)
            self.losses.append(epoch_loss.detach())

            if report_every_epochs > 0 and epoch % report_every_epochs == 0:
                print(f"Epoch {epoch}: loss {epoch_loss}")
        if report_every_epochs >= 0:
            print(f"Final loss {epoch_loss}")

    def infer(self, input, samples):
        def sample_single(model, params, stds, buffers, input):
            perturbed_params = []
            for param, std in zip(params, stds):
                noise = torch.randn_like(param, device=param.device)
                perturbed_params.append(param + noise * std)
            return model(perturbed_params, buffers, input)
        param_stds = vogn_stds(self.optim_state)
        o = vmap(sample_single, in_dims=(None, None, None, None, 0), randomness="different")(self.model, self.params, param_stds, self.buffs, input.expand(samples, *input.shape))
        return o.squeeze(1)

    def forward(self, input, samples=1):
        return self.infer(input, samples)

    def all_losses(self):
        return [self.losses]



############################# iVON #################################
def ivon_init(parameters, N, init):
    states = []
    for param in parameters:
        state = {}
        state["lr"] = init["lr"]
        state["betas"] = init["betas"] if "betas" in init else (0.9, 0.999)
        state["prior_prec"] = init["prior_prec"] if "prior_prec" in init else 1
        state["damping"] = init["damping"] if "damping" in init else 1e-3
        state["tempering"] = init["tempering"] if "tempering" in init else 1
        state["augmentation"] = init["augmentation"] if "augmentation" in init else 1

        state["N"] = N * state["augmentation"]
        state["momentum"] = torch.zeros_like(param, device=param.device)
        state["scale"] = None
        state["step"] = 0

        states.append(state)
    return states

def ivon_prepare(parameters, states):
    perturbed_params = []
    for param, state in zip(parameters, states):
        if state["scale"] is not None:
            std = (1 / (state["N"] * state["scale"])).sqrt()
            noise = torch.randn_like(param, device=param.device) * std
        else:
            noise = torch.zeros_like(param)
        perturbed_params.append(param + noise)
    return perturbed_params, noise

def ivon_step(parameters, grads, states, noise):
    new_parameters = []
    with torch.no_grad():
        for grad, param, state in zip(grads, parameters, states):
            state["step"] += 1
            t = state["step"]
            beta1, beta2 = state["betas"]
            delta = state["prior_prec"] / state["N"]

            grad = grad.mean(dim=0)

            if state["scale"] is None:
                state["scale"] = grad + delta # We treat the first batch as our initialization batch
                new_parameters.append(param)
            else:
                state["momentum"] = beta1 * state["momentum"] + (1 - beta1) * (grad + delta * param)
                update = state["lr"] * state["momentum"] / state["scale"] * (1 - beta2**t) / (1 - beta1**t)
                new_param = param - update

                g_s = state["N"] * noise.mean(dim=0) * grad + delta - state["scale"]
                state["scale"] = state["scale"] + (1 - beta2) * g_s + 0.5 * (1 - beta2)**2 * g_s / state["scale"] * g_s
                
                new_parameters.append(new_param)
    print(new_parameters)
    return new_parameters, states

def ivon_stds(states):
    stds = []
    for state in states:
        stds.append((1 / (state["N"] * state["scale"])).sqrt())
    return stds


class iVONModule(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.model, self.params, self.buffs = make_functional_with_buffers(generate_model(layers))
        self.optim_state = None
        self.losses = []

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return {
            "params": self.params,
            "buffers": self.buffs,
            "optim_state": self.optim_state,
            "losses": self.losses
        }

    def load_state_dict(self, dict):
        self.params = dict["params"]
        self.buffs = dict["buffers"]
        self.optim_state = dict["optim_state"]
        self.losses = dict["losses"]

    def train_model(self, epochs, loss_fn, optim_params, loader, batch_size, device, report_every_epochs=1, mc_samples=10):
        self.params = [p.to(device) for p in self.params]
        self.buffs = [b.to(device) for b in self.buffs]

        self.optim_state = ivon_init(self.params, len(loader) * batch_size, optim_params)

        def get_loss(parameters, input, target):
            input = input.unsqueeze(0)
            target = target.unsqueeze(0)
            output = self.model(parameters, self.buffs, input)
            loss = loss_fn(output, target)
            return loss

        def run_sample(parameters, optim_state, input, target):
            prep_params, noise = ivon_prepare(parameters, optim_state)
            return grad_and_value(get_loss)(prep_params, input, target), noise

        for epoch in range(epochs):
            epoch_loss = torch.tensor(0, dtype=torch.float)
            for data, target in loader:
                data, target = data.to(device), target.to(device)
                data, target = data.expand(mc_samples, -1, -1), target.expand(mc_samples, -1, -1)
                
                (grads, loss), noise = vmap(run_sample, (None, None, 0, 0), randomness="different")(self.params, self.optim_state, data, target)
                loss = loss.mean()
                self.params, self.optim_state = ivon_step(self.params, grads, self.optim_state, noise)
                
                epoch_loss += loss.cpu().item()
            epoch_loss /= (len(loader) * batch_size)
            self.losses.append(epoch_loss.detach())

            if report_every_epochs > 0 and epoch % report_every_epochs == 0:
                print(f"Epoch {epoch}: loss {epoch_loss}")
        if report_every_epochs >= 0:
            print(f"Final loss {epoch_loss}")

    def infer(self, input, samples):
        def sample_single(model, params, stds, buffers, input):
            perturbed_params = []
            for param, std in zip(params, stds):
                noise = torch.randn_like(param, device=param.device)
                perturbed_params.append(param + noise * std)
            return model(perturbed_params, buffers, input)
        param_stds = ivon_stds(self.optim_state)
        o = vmap(sample_single, in_dims=(None, None, None, None, 0), randomness="different")(self.model, self.params, param_stds, self.buffs, input.expand(samples, -1, -1))
        return o.squeeze(1)

    def forward(self, input, samples=1):
        return self.infer(input, samples)

    def all_losses(self):
        return [self.losses]