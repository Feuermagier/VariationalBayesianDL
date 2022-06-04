import pickle

from training.regresssion import RegressionResults

class UCIResults:
    def __init__(self, method, dataset, results: RegressionResults, time):
        self.method = method
        self.dataset = dataset
        self.time = time
        self.results = results

    def store(self, filename):
        with open(filename, "wb+") as file:
            pickle.dump(self, file)

    def load(filename):
        with open(filename, "rb") as file:
            return pickle.load(file)