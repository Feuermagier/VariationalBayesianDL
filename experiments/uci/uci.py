import sys
sys.path.append("../../")

import torch
import matplotlib.pyplot as plt
import time

from experiments.base.uci import UCIDatasets
from experiments.uci.results import UCIResults

from cw2.cw_data import cw_logging
from cw2 import experiment, cw_error, cluster_work

from training.util import adam, nll_loss, plot_losses
from training.swag import SwagModel
from training.bbb import BBBModel, GaussianPrior
from training.ensemble import Ensemble
from training.pp import MAP
from training.regresssion import RegressionResults, plot_calibration


def run(device, config, out_path, log):

    dataset = UCIDatasets(config["dataset"], config["data_path"],
                          test_percentage=config["test_percentage"], normalize=True, subsample=1)

    if config["gap"] is True:
        loaders = [(torch.utils.data.DataLoader(split[0], config["batch_size"], shuffle=False),
            torch.utils.data.DataLoader(split[1], config["batch_size"], shuffle=False)) for split in dataset.gap_splits]
    else:
        loaders = [(torch.utils.data.DataLoader(dataset.train_set, config["batch_size"], shuffle=False),
            torch.utils.data.DataLoader(dataset.test_set, config["batch_size"], shuffle=False))]

    for i, (trainloader, testloader) in enumerate(loaders):

        init_var = torch.tensor(config["init_var"]).to(device)
        model = config["model"]

        before = time.time()
        if model == "map":
            trained_model = run_map(device, trainloader,
                                    dataset.in_dim, init_var, config, out_path)
        elif model == "ensemble":
            trained_model = run_ensemble(
                device, trainloader, dataset.in_dim, init_var, config, out_path)
        elif model == "swag":
            trained_model = run_swag(
                device, trainloader, dataset.in_dim, init_var, config, out_path)
        elif model == "multi_swag":
            trained_model = run_multi_swag(
                device, trainloader, dataset.in_dim, init_var, config, out_path)
        elif model == "mc_dropout":
            trained_model = run_mc_dropout(
                device, trainloader, dataset.in_dim, init_var, config, out_path)
        elif model == "multi_mc_dropout":
            trained_model = run_multi_mc_dropout(
                device, trainloader, dataset.in_dim, init_var, config, out_path)
        elif model == "mfvi":
            trained_model = run_mfvi(
                device, trainloader, dataset.in_dim, init_var, config, out_path)
        elif model == "multi-mfvi":
            trained_model = run_multi_mfvi(
                device, trainloader, dataset.in_dim, init_var, config, out_path)
        elif model == "lrvi":
            trained_model = run_lrvi(
                device, trainloader, dataset.in_dim, init_var, config, out_path)
        else:
            raise ValueError(f"Unknown model type '{model}'")

        after = time.time()
        log.info(f"Time: {after - before}s")

        torch.save(trained_model.state_dict(), out_path + f"model_{i}.tar")

        results = RegressionResults(testloader, model, trained_model.infer,
                                    config["eval_samples"], device, target_mean=dataset.target_mean, target_std=dataset.target_std)
        log.info(f"Avg LML: {results.average_lml}")
        log.info(f"Mean MSE: {results.mean_mse}")
        log.info(f"MSE of Means: {results.mse_of_means}")
        log.info(f"QCE: {results.qce}")

        UCIResults(model, config["dataset"], results, after - before).store(out_path + f"results_{i}.pyc")

        # Plot loss
        fig, ax = plt.subplots()
        plot_losses(model, trained_model.all_losses(), ax)
        fig.set_tight_layout(True)
        fig.savefig(out_path + f"loss_{i}.pdf")

        # Plot calibration
        fig, ax = plt.subplots()
        plot_calibration(None, results, ax, include_text=True)
        fig.set_tight_layout(True)
        fig.savefig(out_path + f"reliability_{i}.pdf")


def run_map(device, trainloader, in_dim, init_var, config, model_out_path):
    layers = [
        ("fc", (in_dim, 50)),
        ("relu", ()),
        ("fc", (50, 1)),
        ("gauss", (init_var, True))
    ]

    model = MAP(layers)
    model.train_model(config["epochs"], nll_loss, adam(
        config["lr"]), trainloader, config["batch_size"], device, report_every_epochs=1)
    
    return model


def run_ensemble(device, trainloader, in_dim, init_var, config, model_out_path):
    members = config["members"]
    layers = [
        ("fc", (in_dim, 50)),
        ("relu", ()),
        ("fc", (50, 1)),
        ("gauss", (init_var, True))
    ]

    model = Ensemble([MAP(layers) for _ in range(members)])
    model.train_model(config["epochs"], nll_loss, adam(
        config["lr"]), trainloader, config["batch_size"], device, report_every_epochs=1)
    torch.save(model.state_dict(), model_out_path + f"ensemble-{members}.tar")
    return model


def run_swag(device, trainloader, in_dim, init_var, config, model_out_path):
    layers = [
        ("fc", (in_dim, 50)),
        ("relu", ()),
        ("fc", (50, 1)),
        ("gauss", (init_var, True))
    ]

    swag_config = config["swag_config"]

    model = SwagModel(layers, swag_config)
    model.train_model(config["epochs"], nll_loss, adam(
        config["lr"]), trainloader, config["batch_size"], device, report_every_epochs=1)
    torch.save(model.state_dict(), model_out_path + "swag.tar")
    return model


def run_multi_swag(device, trainloader, in_dim, init_var, config, model_out_path):
    members = config["members"]
    layers = [
        ("fc", (in_dim, 50)),
        ("relu", ()),
        ("fc", (50, 1)),
        ("gauss", (init_var, True))
    ]

    swag_config = config["swag_config"]

    model = Ensemble([SwagModel(layers, swag_config) for _ in range(members)])
    model.train_model(config["epochs"], nll_loss, adam(
        config["lr"]), trainloader, config["batch_size"], device, report_every_epochs=1)
    torch.save(model.state_dict(), model_out_path +
               f"multi_swag_{members}.tar")
    return model


def run_mc_dropout(device, trainloader, in_dim, init_var, config, model_out_path):
    p = config["p"]
    layers = [
        ("fc", (in_dim, 50)),
        ("dropout", (p,)),
        ("relu", ()),
        ("fc", (50, 1)),
        ("gauss", (init_var, True))
    ]

    model = MAP(layers)
    model.train_model(config["epochs"], nll_loss, adam(
        config["lr"]), trainloader, config["batch_size"], device, report_every_epochs=1)
    torch.save(model.state_dict(), model_out_path + "mc_dropout.tar")
    return model


def run_multi_mc_dropout(device, trainloader, in_dim, init_var, config, model_out_path):
    members = config["members"]
    p = config["p"]
    layers = [
        ("fc", (in_dim, 50)),
        ("dropout", (p,)),
        ("relu", ()),
        ("fc", (50, 1)),
        ("gauss", (init_var, True))
    ]

    model = Ensemble([MAP(layers) for _ in range(members)])
    model.train_model(config["epochs"], nll_loss, adam(
        config["lr"]), trainloader, config["batch_size"], device, report_every_epochs=1)
    torch.save(model.state_dict(), model_out_path +
               f"multi_mc_dropout_{members}.tar")
    return model


def run_mfvi(device, trainloader, in_dim, init_var, config, model_out_path):
    prior = GaussianPrior(0, 1)
    layers = [
        ("v_fc", (in_dim, 50, prior, {"rho_init": -3})),
        ("relu", ()),
        ("v_fc", (50, 1, prior, {"rho_init": -3})),
        ("gauss", (init_var, True))
    ]

    model = BBBModel(layers)
    model.train_model(config["epochs"], nll_loss, adam(config["lr"]), trainloader, config["batch_size"],
                      device, mc_samples=config["mc_samples"], kl_rescaling=config["kl_rescaling"], report_every_epochs=1)
    torch.save(model.state_dict(), model_out_path + f"mfvi.tar")
    return model

def run_lrvi(device, trainloader, in_dim, init_var, config, model_out_path):
    k = config["k"]
    layers = [
        ("vlr_fc", (in_dim, 50, k, 1, {"rho_init": -3})),
        ("relu", ()),
        ("vlr_fc", (50, 1, k, 1, {"rho_init": -3})),
        ("gauss", (init_var, True))
    ]

    model = BBBModel(layers)
    model.train_model(config["epochs"], nll_loss, adam(config["lr"]), trainloader, config["batch_size"],
                      device, mc_samples=config["mc_samples"], kl_rescaling=config["kl_rescaling"], report_every_epochs=1)
    torch.save(model.state_dict(), model_out_path + f"mfvi.tar")
    return model

def run_multi_mfvi(device, trainloader, in_dim, init_var, config, model_out_path):
    members = config["members"]
    prior = GaussianPrior(0, 1)
    layers = [
        ("v_fc", (in_dim, 50, prior, {"rho_init": -3})),
        ("relu", ()),
        ("v_fc", (50, 1, prior, {"rho_init": -3})),
        ("gauss", (init_var, True))
    ]

    model = Ensemble([BBBModel(layers) for _ in range(members)])
    model.train_model(config["epochs"], nll_loss, adam(config["lr"]), trainloader, config["batch_size"],
                      device, mc_samples=config["mc_samples"], kl_rescaling=config["kl_rescaling"], report_every_epochs=1)
    torch.save(model.state_dict(), model_out_path + f"mfvi.tar")
    return model

####################### CW2 #####################################


class UCIExperiment(experiment.AbstractExperiment):
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
    cw = cluster_work.ClusterWork(UCIExperiment)
    cw.run()
