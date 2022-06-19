import pickle

from training.calibration import ClassificationCalibrationResults


class CIFARResults:
    def __init__(self, method, dataset, accuracy, log_likelihood, likelihood, calibration_results: ClassificationCalibrationResults, time, losses):
        self.method = method
        self.dataset = dataset
        self.accuracy = accuracy
        self.log_likelihood = log_likelihood
        self.likelihood = likelihood
        self.time = time
        self.calibration_results = calibration_results
        self.losses = losses

    def store(self, filename):
        with open(filename, "wb+") as file:
            pickle.dump(self, file)

    def load(filename):
        with open(filename, "rb") as file:
            return pickle.load(file)