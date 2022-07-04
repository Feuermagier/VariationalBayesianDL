import pickle

from training.regresssion import RegressionResults

class WeatherResults:
    def __init__(self, method, dataset, results: RegressionResults, time, losses):
        self.method = method
        self.dataset = dataset
        self.time = time
        self.results = results
        self.losses = losses

    def store(self, filename):
        with open(filename, "wb+") as file:
            pickle.dump(self, file)

    def load(filename):
        with open(filename, "rb") as file:
            return pickle.load(file)