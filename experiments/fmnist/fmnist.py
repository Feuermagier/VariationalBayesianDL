import sys
sys.path.append("../../")

import torch
import time

from cw2 import experiment, cw_error, cluster_work
from cw2.cw_data import cw_logging

from experiments.base import mnist
import experiments.base.multiclass_classification as exp
from training.util import sgd
from training.pp import PointPredictor
from training.ensemble import Ensemble
from training.bbb import BBBModel
from training.swag import SwagModel

def run(device, config, out_path, log):
    torch.manual_seed(42)
    trainloader = mnist.fashion_trainloader(config["data_path"], config["batch_size"])
    if config["eval"] == "normal":
        testloader = mnist.fashion_testloader(config["data_path"], config["batch_size"])
    elif config["eval"] == "corrupted":
        testloader = mnist.corrupted_fashion_testloader(config["data_path"], config["batch_size"])
    else:
        raise ValueError(f"Unknown eval dataset '{config['eval']}'")

    model = config["model"]

    before = time.time()
    if model == "map":
        trained_model = run_map(device, trainloader, config, out_path)
    elif model == "ensemble":
        trained_model = run_ensemble(device, trainloader, config, out_path)
    elif model == "swag":
        trained_model = run_swag(device, trainloader, config, out_path)
    elif model == "multi_swag":
        trained_model = run_multi_swag(device, trainloader, config, out_path)
    elif model == "mc_dropout":
        trained_model = run_mc_dropout(device, trainloader, config, out_path)
    elif model == "multi_mc_dropout":
        trained_model = run_mc_dropout(device, trainloader, config, out_path)
    else:
        raise ValueError(f"Unknown model type '{model}'")
    
    after = time.time()
    log.info(f"Time: {after - before}s")

    exp.eval_model(model, trained_model, 1000, testloader, device, out_path, log)

def run_map(device, trainloader, config, model_out_path):
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
        ("fc", (84, 10)),
        ("logsoftmax", ())
    ]

    model = PointPredictor(layers)
    model.train_model(config["epochs"], torch.nn.NLLLoss(), sgd(config["lr"]), trainloader, config["batch_size"], device, report_every_epochs=1)
    torch.save(model.state_dict(), model_out_path + "map.tar")
    return model

def run_ensemble(device, trainloader, config, model_out_path):
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
        ("fc", (84, 10)),
        ("logsoftmax", ())
    ]

    model = Ensemble([PointPredictor(layers) for _ in range(members)])
    model.train_model(config["epochs"], torch.nn.NLLLoss(), sgd(config["lr"]), trainloader, config["batch_size"], device, report_every_epochs=1)
    torch.save(model.state_dict(), model_out_path + f"ensemble-{members}.tar")
    return model

def run_swag(device, trainloader, config, model_out_path):
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
        ("fc", (84, 10)),
        ("logsoftmax", ())
    ]

    swag_config = config["swag_config"]

    model = SwagModel(layers, swag_config)
    model.train_model(config["epochs"], torch.nn.NLLLoss(), sgd(config["lr"]), trainloader, config["batch_size"], device, report_every_epochs=1)
    torch.save(model.state_dict(), model_out_path + "swag.tar")
    return model

def run_multi_swag(device, trainloader, config, model_out_path):
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
        ("fc", (84, 10)),
        ("logsoftmax", ())
    ]

    swag_config = config["swag_config"]

    model = Ensemble([SwagModel(layers, swag_config) for _ in range(members)])
    model.train_model(config["epochs"], torch.nn.NLLLoss(), sgd(config["lr"]), trainloader, config["batch_size"], device, report_every_epochs=1)
    torch.save(model.state_dict(), model_out_path + f"multi_swag_{members}.tar")
    return model

def run_mc_dropout(device, trainloader, config, model_out_path):
    p = config["dropout_p"]
    layers = [
        ("conv", (1, 6, 5)),
        ("relu", ()),
        ("pool", 2),
        ("conv", (6, 16, 5)),
        ("relu", ()),
        ("pool", 2),
        ("flatten", ()),
        ("fc", (16 * 4 * 4, 120)),
        ("dropout", (p,))
        ("relu", ()),
        ("fc", (120, 84)),
        ("dropout", (p,))
        ("relu", ()),
        ("fc", (84, 10)),
        ("logsoftmax", ())
    ]

    model = PointPredictor(layers)
    model.train_model(config["epochs"], torch.nn.NLLLoss(), sgd(config["lr"]), trainloader, config["batch_size"], device, report_every_epochs=1)
    torch.save(model.state_dict(), model_out_path + "mc_dropout.tar")
    return model

def run_multi_mc_dropout(device, trainloader, config, model_out_path):
    members = config["members"]
    p = config["dropout_p"]
    layers = [
        ("conv", (1, 6, 5)),
        ("relu", ()),
        ("pool", 2),
        ("conv", (6, 16, 5)),
        ("relu", ()),
        ("pool", 2),
        ("flatten", ()),
        ("fc", (16 * 4 * 4, 120)),
        ("dropout", (p,))
        ("relu", ()),
        ("fc", (120, 84)),
        ("dropout", (p,))
        ("relu", ()),
        ("fc", (84, 10)),
        ("logsoftmax", ())
    ]

    model = Ensemble([PointPredictor(layers) for _ in range(members)])
    model.train_model(config["epochs"], torch.nn.NLLLoss(), sgd(config["lr"]), trainloader, config["batch_size"], device, report_every_epochs=1)
    torch.save(model.state_dict(), model_out_path + f"multi_mc_dropout_{members}.tar")
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

        run(device, config["params"], config["_rep_log_path"], l)
    
    def finalize(self, surrender: cw_error.ExperimentSurrender = None, crash: bool = False):
        pass

if __name__ == "__main__":
    cw = cluster_work.ClusterWork(FashionMNISTExperiment)
    cw.run()