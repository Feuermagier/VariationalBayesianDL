import sys
sys.path.append("../../")

import torch
import time

from cw2 import experiment, cw_error, cluster_work
from cw2.cw_data import cw_logging

from experiments.base import cifar
from experiments.cifar.results import CIFARResults
import experiments.base.multiclass_classification as exp
from training.util import sgd, lr_scheduler, wilson_scheduler
from training.pp import MAP
from training.ensemble import Ensemble
from training.bbb import BBBModel, GaussianPrior
from training.swag import SwagModel

def run(device, config, out_path, log):
    class_exclusion = config["classes"]
    if class_exclusion != []:
        log.info(f"Excluding classes {class_exclusion} from training")
    trainloader = cifar.cifar10_trainloader(config["data_path"], config["batch_size"], exclude_classes=class_exclusion)

    model = config["model"]

    before = time.time()
    if model == "map":
        trained_model = run_map(device, trainloader, config)
    elif model == "ensemble":
        trained_model = run_ensemble(device, trainloader, config)
    elif model == "swag":
        trained_model = run_swag(device, trainloader, config)
    elif model == "multi_swag":
        trained_model = run_multi_swag(device, trainloader, config)
    elif model == "mcd":
        trained_model = run_mcd(device, trainloader, config)
    elif model == "multi_mcd":
        trained_model = run_multi_mcd(device, trainloader, config)
    elif model == "bbb":
        trained_model = run_bbb(device, trainloader, config)
    elif model == "multi_bbb":
        trained_model = run_multi_bbb(device, trainloader, config)
    else:
        raise ValueError(f"Unknown model type '{model}'")
    
    after = time.time()
    log.info(f"Time: {after - before}s")

    torch.save(trained_model.state_dict(), out_path + "model.tar")

    classes = [i for i in range(10) if i not in class_exclusion] if class_exclusion != [] else []

    if class_exclusion != []:
        log.info(f"Evaluating only on classes {class_exclusion}")
    testloader = cifar.cifar10_testloader(config["data_path"], config["batch_size"], exclude_classes=classes)
    acc, log_likelihood, likelihood, cal_res = exp.eval_model(trained_model, config["eval_samples"], testloader, device, "normal", log)
    CIFARResults(model, "standard", acc, log_likelihood, likelihood, cal_res, after - before, trained_model.all_losses()).store(out_path + "results_normal.pyc")
    del testloader

    for i in config["intensities"]:
        testloader = cifar.cifar10_corrupted_testloader(config["data_path"], i, config["batch_size"])
        acc, log_likelihood, likelihood, cal_res = exp.eval_model(trained_model, config["eval_samples"], testloader, device, f"C({i})", log)
        CIFARResults(model, f"C({i})", acc, log_likelihood, likelihood, cal_res, after - before, trained_model.all_losses()).store(out_path + f"results_{i}.pyc")
        del testloader

    if config["stl10"]:
        testloader = cifar.stl10_testloader(config["data_path"], config["batch_size"])
        acc, log_likelihood, likelihood, cal_res = exp.eval_model(trained_model, config["eval_samples"], testloader, device, f"STL10", log)
        CIFARResults(model, f"STL10", acc, log_likelihood, likelihood, cal_res, after - before, trained_model.all_losses()).store(out_path + f"results_stl10.pyc")
        del testloader

def optimizer(config):
    return sgd(config["lr"], weight_decay=config["weight_decay"], momentum=config["momentum"], nesterov=config["nesterov"])

def schedule(config):
    # return lr_scheduler(config["lr_milestones"], config["lr_decay"])
    return wilson_scheduler(config["epochs"], config["lr"], None)

def run_map(device, trainloader, config):
    layers = [
        ("preresnet-20", (32, 3, 10)),
        ("logsoftmax", ())
    ]

    model = MAP(layers)
    model.train_model(config["epochs"], torch.nn.NLLLoss(), optimizer(config), trainloader, config["batch_size"], device, scheduler_factory=schedule(config), report_every_epochs=1)
    return model

def run_ensemble(device, trainloader, config):
    members = config["members"]
    layers = [
        ("preresnet-20", (32, 3, 10)),
        ("logsoftmax", ())
    ]

    model = Ensemble([MAP(layers) for _ in range(members)])
    model.train_model(config["epochs"], torch.nn.NLLLoss(), optimizer(config), trainloader, config["batch_size"], device, scheduler_factory=schedule(config), report_every_epochs=1)
    return model

def run_swag(device, trainloader, config):
    layers = [
        ("preresnet-20", (32, 3, 10)),
        ("logsoftmax", ())
    ]

    swag_config = config["swag_config"]

    model = SwagModel(layers, swag_config)
    scheduler = wilson_scheduler(swag_config["start_epoch"], config["lr"], swag_config["lr"])
    model.train_model(config["epochs"], torch.nn.NLLLoss(), optimizer(config), trainloader, config["batch_size"], device, scheduler_factory=scheduler, report_every_epochs=1)
    return model

def run_multi_swag(device, trainloader, config):
    members = config["members"]
    layers = [
        ("preresnet-20", (32, 3, 10)),
        ("logsoftmax", ())
    ]

    swag_config = config["swag_config"]

    model = Ensemble([SwagModel(layers, swag_config) for _ in range(members)])
    scheduler = wilson_scheduler(swag_config["start_epoch"], config["lr"], swag_config["lr"])
    model.train_model(config["epochs"], torch.nn.NLLLoss(), optimizer(config), trainloader, config["batch_size"], device, scheduler_factory=scheduler, report_every_epochs=1)
    return model

def run_mcd(device, trainloader, config):
    layers = [
        ("drop-preresnet-20", (32, 3, 10, config["p"])),
        ("logsoftmax", ())
    ]

    model = MAP(layers)
    model.train_model(config["epochs"], torch.nn.NLLLoss(), optimizer(config), trainloader, config["batch_size"], device, scheduler_factory=schedule(config), report_every_epochs=1)
    return model

def run_multi_mcd(device, trainloader, config):
    members = config["members"]
    layers = [
        ("drop-preresnet-20", (32, 3, 10, config["p"])),
        ("logsoftmax", ())
    ]

    model = Ensemble([MAP(layers) for _ in range(members)])
    model.train_model(config["epochs"], torch.nn.NLLLoss(), optimizer(config), trainloader, config["batch_size"], device, scheduler_factory=schedule(config), report_every_epochs=1)
    return model

def run_bbb(device, trainloader, config):
    prior = GaussianPrior(torch.tensor(config["prior_mean"]), torch.tensor(config["prior_std"]))
    layers = [
        ("variational-preresnet-20", (32, 3, 10, prior)),
        ("logsoftmax", ())
    ]

    model = BBBModel(layers)
    model.train_model(config["epochs"], torch.nn.NLLLoss(), optimizer(config), trainloader, config["batch_size"], device, scheduler_factory=schedule(config), mc_samples=config["mc_samples"], kl_rescaling=config["kl_rescaling"], report_every_epochs=1)
    return model

def run_multi_bbb(device, trainloader, config):
    members = config["members"]
    prior = GaussianPrior(torch.tensor(config["prior_mean"]), torch.tensor(config["prior_std"]))
    layers = [
        ("variational-preresnet-20", (32, 3, 10, prior)),
        ("logsoftmax", ())
    ]

    model = Ensemble([BBBModel(layers) for _ in range(members)])
    model.train_model(config["epochs"], torch.nn.NLLLoss(), optimizer(config), trainloader, config["batch_size"], device, scheduler_factory=schedule(config), mc_samples=config["mc_samples"], kl_rescaling=config["kl_rescaling"], report_every_epochs=1)
    return model

####################### CW2 #####################################
class FashionMNISTExperiment(experiment.AbstractExperiment):
    def initialize(self, config: dict, rep: int, logger: cw_logging.LoggerArray) -> None:
        pass

    def run(self, config: dict, rep: int, logger: cw_logging.LoggerArray) -> None:
        l = cw_logging.getLogger()
        l.info(config["params"])
        if torch.cuda.is_available():
            l.info("Using the GPU")
            device = torch.device("cuda")
        else:
            l.info("Using the CPU")
            device = torch.device("cpu")

        torch.manual_seed(rep * 42)

        run(device, config["params"], config["_rep_log_path"], l)
    
    def finalize(self, surrender: cw_error.ExperimentSurrender = None, crash: bool = False):
        pass

if __name__ == "__main__":
    cw = cluster_work.ClusterWork(FashionMNISTExperiment)
    cw.run()
