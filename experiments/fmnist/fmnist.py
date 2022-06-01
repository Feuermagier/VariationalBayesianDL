import sys
sys.path.append("../../")

import torch

from cw2 import experiment, cw_error, cluster_work
from cw2.cw_data import cw_logging

from experiments.base import mnist
from training.util import sgd
from training.pp import PointPredictor

def run(device, config, out_path):
    torch.manual_seed(42)
    trainloader = mnist.fashion_trainloader(config["data_path"], config["batch_size"])
    testloader = mnist.fashion_testloader(config["data_path"], config["batch_size"])
    corrupted_testloader = mnist.corrupted_fashion_testloader(config["data_path"], config["batch_size"])

    model = config["model"]

    if model == "map":
        run_map(device, trainloader, config, out_path)


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

    map = PointPredictor(layers)
    map.train_model(config["epochs"], torch.nn.NLLLoss(), sgd(config["lr"]), trainloader, config["batch_size"], device, report_every_epochs=1)
    torch.save(map.state_dict(), model_out_path + "map.tar")


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

        run(device, config["params"], config["_rep_log_path"])
    
    def finalize(self, surrender: cw_error.ExperimentSurrender = None, crash: bool = False):
        pass

if __name__ == "__main__":
    cw = cluster_work.ClusterWork(FashionMNISTExperiment)
    cw.run()