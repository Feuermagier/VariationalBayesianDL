import sys
sys.path.append("../../")

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time

from experiments.base.shifts import WeatherShiftsDataset
from experiments.weather.results import WeatherResults

from cw2.cw_data import cw_logging
from cw2 import experiment, cw_error, cluster_work

from training.util import adam, nll_loss, plot_losses, EarlyStopper
from training.swag import SwagModel
from training.bbb import BBBModel, GaussianPrior
from training.ensemble import Ensemble
from training.pp import MAP
from training.regresssion import RegressionResults, plot_calibration
from training.sgld import SGLDModule, sgld


def run(device, config, out_path, log):

    dataset = WeatherShiftsDataset(config["data_path"])

    trainloader = dataset.trainloader(config["batch_size"])
    valloader = dataset.in_valloader(1000)

    init_std = torch.tensor(config["init_std"]).to(device)
    model = config["model"]

    def validate(model):
        with torch.no_grad():
            loss = 0
            for data, target in valloader:
                output = model(data, 100).mean(dim=0)
                loss += F.mse_loss(output[...,0], target).detach().item()
            return loss / len(valloader)

    es_config = config["early_stopping"]
    es = EarlyStopper(validate, es_config["interval"], es_config["delta"], es_config["patience"])

    before = time.time()
    if model == "map":
        trained_model = run_map(device, trainloader,
                                init_std, es, config)
    elif model == "ensemble":
        trained_model = run_ensemble(
            device, trainloader, init_std, config)
    elif model == "swag":
        trained_model = run_swag(
            device, trainloader, init_std, config)
    elif model == "multi_swag":
        trained_model = run_multi_swag(
            device, trainloader, init_std, config)
    elif model == "mc_dropout":
        trained_model = run_mc_dropout(
            device, trainloader, init_std, config)
    elif model == "multi_mc_dropout":
        trained_model = run_multi_mc_dropout(
            device, trainloader, init_std, config)
    elif model == "mfvi":
        trained_model = run_mfvi(
            device, trainloader, init_std, config)
    elif model == "multi_mfvi":
        trained_model = run_multi_mfvi(
            device, trainloader, init_std, config)
    else:
        raise ValueError(f"Unknown model type '{model}'")

    after = time.time()
    log.info(f"Time: {after - before}s")

    torch.save(trained_model.state_dict(), out_path + f"model.tar")

    # Plot loss
    fig, ax = plt.subplots()
    plot_losses(model, trained_model.all_losses(), ax)
    fig.set_tight_layout(True)
    fig.savefig(out_path + f"loss.pdf")

    # Eval in
    testloader = dataset.in_testloader(128)
    results = RegressionResults(testloader, model, trained_model.infer,
                                config["eval_samples"], device, target_mean=dataset.target_mean, target_std=dataset.target_std)
    log.info(f"In Avg LML: {results.average_lml}")
    log.info(f"In Mean MSE: {results.mean_mse}")
    log.info(f"In MSE of Means: {results.mse_of_means}")
    log.info(f"In QCE: {results.qce}")
    WeatherResults(model, "in", results, after - before).store(out_path + f"results_in.pyc")

    # Eval out
    testloader = dataset.out_testloader(128)
    results = RegressionResults(testloader, model, trained_model.infer,
                                config["eval_samples"], device, target_mean=dataset.target_mean, target_std=dataset.target_std)
    log.info(f"Out Avg LML: {results.average_lml}")
    log.info(f"Out Mean MSE: {results.mean_mse}")
    log.info(f"Out MSE of Means: {results.mse_of_means}")
    log.info(f"Out QCE: {results.qce}")
    WeatherResults(model, "out", results, after - before).store(out_path + f"results_out.pyc")


def run_map(device, trainloader, init_std, es, config):
    layers = [
        ("fc", (123, 250)),
        ("relu", ()),
        ("fc", (250, 250)),
        ("relu", ()),
        ("fc", (250, 250)),
        ("relu", ()),
        ("fc", (250, 250)),
        ("relu", ()),
        ("fc", (250, 1)),
        ("gauss", (init_std, True)),
    ]

    es = EarlyStopper()

    model = MAP(layers)
    model.train_model(config["epochs"], nll_loss, adam(
        config["lr"]), trainloader, config["batch_size"], device, early_stopping=es, report_every_epochs=1)
    return model


def run_ensemble(device, trainloader, init_std, config):
    members = config["members"]
    layers = [
        ("fc", (123, 250)),
        ("relu", ()),
        ("fc", (250, 250)),
        ("relu", ()),
        ("fc", (250, 1)),
        ("gauss", (init_std, True)),
    ]

    model = Ensemble([MAP(layers) for _ in range(members)])
    model.train_model(config["epochs"], nll_loss, adam(
        config["lr"]), trainloader, config["batch_size"], device, report_every_epochs=1)
    return model


def run_swag(device, trainloader, init_std, config):
    layers = [
        ("fc", (123, 250)),
        ("relu", ()),
        ("fc", (250, 250)),
        ("relu", ()),
        ("fc", (250, 1)),
        ("gauss", (init_std, True)),
    ]

    swag_config = config["swag_config"]

    model = SwagModel(layers, swag_config)
    model.train_model(config["epochs"], nll_loss, adam(
        config["lr"]), trainloader, config["batch_size"], device, report_every_epochs=1)
    return model


def run_multi_swag(device, trainloader, init_std, config):
    members = config["members"]
    layers = [
        ("fc", (123, 250)),
        ("relu", ()),
        ("fc", (250, 250)),
        ("relu", ()),
        ("fc", (250, 1)),
        ("gauss", (init_std, True)),
    ]

    swag_config = config["swag_config"]

    model = Ensemble([SwagModel(layers, swag_config) for _ in range(members)])
    model.train_model(config["epochs"], nll_loss, adam(
        config["lr"]), trainloader, config["batch_size"], device, report_every_epochs=1)
    return model


def run_mc_dropout(device, trainloader, init_std, config):
    p = config["p"]
    layers = [
        ("fc", (123, 250)),
        ("relu", ()),
        ("fc", (250, 250)),
        ("relu", ()),
        ("fc", (250, 1)),
        ("gauss", (init_std, True)),
    ]

    model = MAP(layers)
    model.train_model(config["epochs"], nll_loss, adam(
        config["lr"]), trainloader, config["batch_size"], device, report_every_epochs=1)
    return model


def run_multi_mc_dropout(device, trainloader, init_std, config):
    members = config["members"]
    p = config["p"]
    layers = [
        ("fc", (123, 250)),
        ("relu", ()),
        ("fc", (250, 250)),
        ("relu", ()),
        ("fc", (250, 1)),
        ("gauss", (init_std, True)),
    ]

    model = Ensemble([MAP(layers) for _ in range(members)])
    model.train_model(config["epochs"], nll_loss, adam(
        config["lr"]), trainloader, config["batch_size"], device, report_every_epochs=1)
    return model


def run_mfvi(device, trainloader, init_std, config):
    prior = GaussianPrior(0, 1)
    layers = [
        ("v_fc", (123, 250, prior)),
        ("relu", ()),
        ("v_fc", (250, 250, prior)),
        ("relu", ()),
        ("v_fc", (250, 1, prior)),
        ("gauss", (init_std, True)),
    ]

    model = BBBModel(layers)
    model.train_model(config["epochs"], nll_loss, adam(config["lr"]), trainloader, config["batch_size"],
                      device, mc_samples=config["mc_samples"], kl_rescaling=config["kl_rescaling"], report_every_epochs=1)
    return model

def run_multi_mfvi(device, trainloader, init_std, config):
    members = config["members"]
    prior = GaussianPrior(0, 1)
    layers = [
        ("v_fc", (123, 250, prior)),
        ("relu", ()),
        ("v_fc", (250, 250, prior)),
        ("relu", ()),
        ("v_fc", (250, 1, prior)),
        ("gauss", (init_std, True)),
    ]

    model = Ensemble([BBBModel(layers) for _ in range(members)])
    model.train_model(config["epochs"], nll_loss, adam(config["lr"]), trainloader, config["batch_size"],
                      device, mc_samples=config["mc_samples"], kl_rescaling=config["kl_rescaling"], report_every_epochs=1)
    return model

####################### CW2 #####################################


class WeatherExperiment(experiment.AbstractExperiment):
    def initialize(self, config: dict, rep: int, logger: cw_logging.LoggerArray) -> None:
        pass

    def run(self, config: dict, rep: int, logger: cw_logging.LoggerArray) -> None:
        l = cw_logging.getLogger()
        l.info(config["params"])
        l.info("Using the CPU")
        device = torch.device("cpu")

        torch.manual_seed(rep * 42)

        run(device, config["params"], config["_rep_log_path"], l)

    def finalize(self, surrender: cw_error.ExperimentSurrender = None, crash: bool = False):
        pass


if __name__ == "__main__":
    cw = cluster_work.ClusterWork(WeatherExperiment)
    cw.run()
