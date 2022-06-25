import sys
sys.path.append("../../")

import torch
import time

from cw2 import experiment, cw_error, cluster_work
from cw2.cw_data import cw_logging

from experiments.base import cifar
from experiments.cifar.results import CIFARResults
import experiments.base.multiclass_classification as exp
from training.util import sgd
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
    else:
        raise ValueError(f"Unknown model type '{model}'")
    
    after = time.time()
    log.info(f"Time: {after - before}s")

    #torch.save(trained_model.state_dict(), out_path + "model.tar")

    classes = [i for i in range(10) if i not in class_exclusion] if class_exclusion != [] else []

    if class_exclusion != []:
        log.info(f"Evaluating only on classes {class_exclusion}")
    testloader = cifar.cifar10_testloader(config["data_path"], config["batch_size"], exclude_classes=classes)
    acc, log_likelihood, likelihood, cal_res = exp.eval_model(trained_model, config["eval_samples"], testloader, device, "normal", log)
    CIFARResults(model, "standard", acc, log_likelihood, likelihood, cal_res, after - before, trained_model.all_losses()).store(out_path + "results_normal.pyc")

    for i in config["intensities"]:
        testloader = cifar.cifar10_corrupted_testloader(config["data_path"], i, config["batch_size"])
        acc, log_likelihood, likelihood, cal_res = exp.eval_model(trained_model, config["eval_samples"], testloader, device, f"C({i})", log)
        CIFARResults(model, f"C({i})", acc, log_likelihood, likelihood, cal_res, after - before, trained_model.all_losses()).store(out_path + f"results_{i}.pyc")

def run_map(device, trainloader, config):
    layers = [
        ("preresnet-20", (32, 3, 10)),
        ("logsoftmax", ())
    ]

    model = MAP(layers)
    model.train_model(config["epochs"], torch.nn.NLLLoss(), sgd(config["lr"], weight_decay=config["weight_decay"]), trainloader, config["batch_size"], device, report_every_epochs=1)
    return model

def run_ensemble(device, trainloader, config):
    members = config["members"]
    layers = [
        ("preresnet-20", (32, 3, 10)),
        ("logsoftmax", ())
    ]

    model = Ensemble([MAP(layers) for _ in range(members)])
    model.train_model(config["epochs"], torch.nn.NLLLoss(), sgd(config["lr"], weight_decay=config["weight_decay"]), trainloader, config["batch_size"], device, report_every_epochs=1)
    return model

def run_swag(device, trainloader, config):
    layers = [
        ("preresnet-20", (32, 3, 10)),
        ("logsoftmax", ())
    ]

    swag_config = config["swag_config"]

    model = SwagModel(layers, swag_config)
    model.train_model(config["epochs"], torch.nn.NLLLoss(), sgd(config["lr"], weight_decay=config["weight_decay"]), trainloader, config["batch_size"], device, report_every_epochs=1)
    return model

def run_multi_swag(device, trainloader, config):
    members = config["members"]
    layers = [
        ("preresnet-20", (32, 3, 10)),
        ("logsoftmax", ())
    ]

    swag_config = config["swag_config"]

    model = Ensemble([SwagModel(layers, swag_config) for _ in range(members)])
    model.train_model(config["epochs"], torch.nn.NLLLoss(), sgd(config["lr"], weight_decay=config["weight_decay"]), trainloader, config["batch_size"], device, report_every_epochs=1)
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
