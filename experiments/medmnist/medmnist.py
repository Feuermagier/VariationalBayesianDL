import sys
sys.path.append("../../")

import torch
import time

from cw2 import experiment, cw_error, cluster_work
from cw2.cw_data import cw_logging

from experiments.base import mnist
from experiments.medmnist.results import MedMNISTResults
import experiments.base.multiclass_classification as exp
from training.util import sgd
from training.pp import MAP
from training.ensemble import Ensemble
from training.bbb import BBBModel, GaussianPrior
from training.swag import SwagModel
from training.vogn import VOGNModule

def run(device, config, out_path, log):
    class_exclusion = config["classes"]
    if class_exclusion != []:
        log.info(f"Excluding classes {class_exclusion} from training")
    dataset = config["dataset"]
    trainloader = mnist.medmnist_trainloader(config["data_path"], dataset, config["batch_size"], exclude_classes=class_exclusion)
    classes = mnist.medmnist_classes[dataset]

    model = config["model"]

    before = time.time()
    if model == "map":
        trained_model = run_map(device, trainloader, classes, config)
    elif model == "ensemble":
        trained_model = run_ensemble(device, trainloader, classes, config)
    elif model == "swag":
        trained_model = run_swag(device, trainloader, classes, config)
    elif model == "multi_swag":
        trained_model = run_multi_swag(device, trainloader, classes, config)
    elif model == "mc_dropout":
        trained_model = run_mc_dropout(device, trainloader, classes, config)
    elif model == "multi_mc_dropout":
        trained_model = run_multi_mc_dropout(device, trainloader, classes, config)
    elif model == "mfvi":
        trained_model = run_mfvi(device, trainloader, classes, config)
    elif model == "multi_mfvi":
        trained_model = run_multi_mfvi(device, trainloader, classes, config)
    elif model == "vogn":
        trained_model = run_vogn(device, trainloader, classes, config)
    elif model == "multi_vogn":
        trained_model = run_multi_vogn(device, trainloader, classes, config)
    else:
        raise ValueError(f"Unknown model type '{model}'")
    
    after = time.time()
    log.info(f"Time: {after - before}s")

    #torch.save(trained_model.state_dict(), out_path + "model.tar")

    classes = [i for i in range(10) if i not in class_exclusion] if class_exclusion != [] else []
    if class_exclusion != []:
        log.info(f"Evaluating only on classes {class_exclusion}")
    testloader = mnist.medmnist_testloader(config["data_path"], dataset, config["batch_size"], exclude_classes=classes)
    acc, log_likelihood, likelihood, cal_res = exp.eval_model(trained_model, config["eval_samples"], testloader, device, dataset, log)
    MedMNISTResults(model, dataset, acc, log_likelihood, likelihood, cal_res, after - before, trained_model.all_losses()).store(out_path + f"results_{dataset}.pyc")


def run_map(device, trainloader, classes, config):
    layers = [
        ("conv", (1, 6, 5)),
        ("relu", ()),
        ("pool", 2),
        ("conv", (6, 16, 5)),
        ("relu", ()),
        ("pool", 2),
        ("flatten", ()),
        ("fc", (16 * 4 * 4, 120)),
        ("relu", ()),
        ("fc", (120, 84)),
        ("relu", ()),
        ("fc", (84, classes)),
        ("logsoftmax", ())
    ]

    model = MAP(layers)
    model.train_model(config["epochs"], torch.nn.NLLLoss(), sgd(config["lr"], weight_decay=config["weight_decay"]), trainloader, config["batch_size"], device, report_every_epochs=1)
    return model

def run_ensemble(device, trainloader, classes, config):
    members = config["members"]
    layers = [
        ("conv", (1, 6, 5)),
        ("relu", ()),
        ("pool", 2),
        ("conv", (6, 16, 5)),
        ("relu", ()),
        ("pool", 2),
        ("flatten", ()),
        ("fc", (16 * 4 * 4, 120)),
        ("relu", ()),
        ("fc", (120, 84)),
        ("relu", ()),
        ("fc", (84, classes)),
        ("logsoftmax", ())
    ]

    model = Ensemble([MAP(layers) for _ in range(members)])
    model.train_model(config["epochs"], torch.nn.NLLLoss(), sgd(config["lr"], weight_decay=config["weight_decay"]), trainloader, config["batch_size"], device, report_every_epochs=1)
    return model

def run_swag(device, trainloader, classes, config):
    layers = [
        ("conv", (1, 6, 5)),
        ("relu", ()),
        ("pool", 2),
        ("conv", (6, 16, 5)),
        ("relu", ()),
        ("pool", 2),
        ("flatten", ()),
        ("fc", (16 * 4 * 4, 120)),
        ("relu", ()),
        ("fc", (120, 84)),
        ("relu", ()),
        ("fc", (84, classes)),
        ("logsoftmax", ())
    ]

    swag_config = config["swag_config"]

    model = SwagModel(layers, swag_config)
    model.train_model(config["epochs"], torch.nn.NLLLoss(), sgd(config["lr"], weight_decay=config["weight_decay"]), trainloader, config["batch_size"], device, report_every_epochs=1)
    return model

def run_multi_swag(device, trainloader, classes, config):
    members = config["members"]
    layers = [
        ("conv", (1, 6, 5)),
        ("relu", ()),
        ("pool", 2),
        ("conv", (6, 16, 5)),
        ("relu", ()),
        ("pool", 2),
        ("flatten", ()),
        ("fc", (16 * 4 * 4, 120)),
        ("relu", ()),
        ("fc", (120, 84)),
        ("relu", ()),
        ("fc", (84, classes)),
        ("logsoftmax", ())
    ]

    swag_config = config["swag_config"]

    model = Ensemble([SwagModel(layers, swag_config) for _ in range(members)])
    model.train_model(config["epochs"], torch.nn.NLLLoss(), sgd(config["lr"], weight_decay=config["weight_decay"]), trainloader, config["batch_size"], device, report_every_epochs=1)
    return model

def run_mc_dropout(device, trainloader, classes, config):
    p = config["p"]
    layers = [
        ("conv", (1, 6, 5)),
        ("dropout", (p,)),
        ("relu", ()),
        ("pool", 2),
        ("conv", (6, 16, 5)),
        ("dropout", (p,)),
        ("relu", ()),
        ("pool", 2),
        ("flatten", ()),
        ("fc", (16 * 4 * 4, 120)),
        ("dropout", (p,)),
        ("relu", ()),
        ("fc", (120, 84)),
        ("dropout", (p,)),
        ("relu", ()),
        ("fc", (84, classes)),
        ("logsoftmax", ())
    ]

    model = MAP(layers)
    model.train_model(config["epochs"], torch.nn.NLLLoss(), sgd(config["lr"], weight_decay=config["weight_decay"]), trainloader, config["batch_size"], device, mc_samples=config["mc_samples"], report_every_epochs=1)
    return model

def run_multi_mc_dropout(device, trainloader, classes, config):
    members = config["members"]
    p = config["p"]
    layers = [
        ("conv", (1, 6, 5)),
        ("dropout", (p,)),
        ("relu", ()),
        ("pool", 2),
        ("conv", (6, 16, 5)),
        ("dropout", (p,)),
        ("relu", ()),
        ("pool", 2),
        ("flatten", ()),
        ("fc", (16 * 4 * 4, 120)),
        ("dropout", (p,)),
        ("relu", ()),
        ("fc", (120, 84)),
        ("dropout", (p,)),
        ("relu", ()),
        ("fc", (84, classes)),
        ("logsoftmax", ())
    ]

    model = Ensemble([MAP(layers) for _ in range(members)])
    model.train_model(config["epochs"], torch.nn.NLLLoss(), sgd(config["lr"], weight_decay=config["weight_decay"]), trainloader, config["batch_size"], device, mc_samples=config["mc_samples"], report_every_epochs=1)
    return model

def run_mfvi(device, trainloader, classes, config):
    prior = GaussianPrior(0, 1)
    layers = [
        ("v_conv", (1, 6, 5, prior, {})),
        ("relu", ()),
        ("pool", 2),
        ("v_conv", (6, 16, 5, prior, {})),
        ("relu", ()),
        ("pool", 2),
        ("flatten", ()),
        ("v_fc", (16 * 4 * 4, 120, prior, {})),
        ("relu", ()),
        ("v_fc", (120, 84, prior, {})),
        ("relu", ()),
        ("v_fc", (84, classes, prior, {})),
        ("logsoftmax", ())
    ]

    model = BBBModel(layers)
    model.train_model(config["epochs"], torch.nn.NLLLoss(), sgd(config["lr"], weight_decay=config["weight_decay"]), trainloader, config["batch_size"], device, mc_samples=config["mc_samples"], kl_rescaling=config["kl_rescaling"], report_every_epochs=1)
    return model

def run_multi_mfvi(device, trainloader, classes, config):
    members = config["members"]
    prior = GaussianPrior(0, 1)
    layers = [
        ("v_conv", (1, 6, 5, prior, {})),
        ("relu", ()),
        ("pool", 2),
        ("v_conv", (6, 16, 5, prior, {})),
        ("relu", ()),
        ("pool", 2),
        ("flatten", ()),
        ("v_fc", (16 * 4 * 4, 120, prior, {})),
        ("relu", ()),
        ("v_fc", (120, 84, prior, {})),
        ("relu", ()),
        ("v_fc", (84, classes, prior, {})),
        ("logsoftmax", ())
    ]

    model = Ensemble([BBBModel(layers) for _ in range(members)])
    model.train_model(config["epochs"], torch.nn.NLLLoss(), sgd(config["lr"], weight_decay=config["weight_decay"]), trainloader, config["batch_size"], device, mc_samples=config["mc_samples"], kl_rescaling=config["kl_rescaling"], report_every_epochs=1)
    return model

def run_vogn(device, trainloader, classes, config):
    layers = [
        ("conv", (1, 6, 5)),
        ("relu", ()),
        ("pool", 2),
        ("conv", (6, 16, 5)),
        ("relu", ()),
        ("pool", 2),
        ("flatten", ()),
        ("fc", (16 * 4 * 4, 120)),
        ("relu", ()),
        ("fc", (120, 84)),
        ("relu", ()),
        ("fc", (84, classes)),
        ("logsoftmax", ())
    ]

    model = VOGNModule(layers)
    model.train_model(config["epochs"], torch.nn.NLLLoss(), config["vogn"], trainloader, config["batch_size"], device, mc_samples=config["mc_samples"], report_every_epochs=1)
    return model

def run_multi_vogn(device, trainloader, classes, config):
    members = config["members"]
    layers = [
        ("conv", (1, 6, 5)),
        ("relu", ()),
        ("pool", 2),
        ("conv", (6, 16, 5)),
        ("relu", ()),
        ("pool", 2),
        ("flatten", ()),
        ("fc", (16 * 4 * 4, 120)),
        ("relu", ()),
        ("fc", (120, 84)),
        ("relu", ()),
        ("fc", (84, classes)),
        ("logsoftmax", ())
    ]

    model = Ensemble([VOGNModule(layers) for _ in range(members)])
    model.train_model(config["epochs"], torch.nn.NLLLoss(), config["vogn"], trainloader, config["batch_size"], device, mc_samples=config["mc_samples"], report_every_epochs=1)
    return model

####################### CW2 #####################################
class MedMNISTExperiment(experiment.AbstractExperiment):
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
    cw = cluster_work.ClusterWork(MedMNISTExperiment)
    cw.run()