import numpy as np
import torch
import pandas as pd
from training import util

class UCIDatasets():
    def __init__(self,  name,  data_path, test_percentage=0.2, normalize=True, subsample=1):
        self.data_path = data_path
        self.name = name

        if self.name == "housing":
            data = pd.read_csv(self.data_path+'UCI/housing.data',
                        header=0, delimiter="\s+").values

        elif self.name == "concrete":
            data = pd.read_excel(self.data_path+'UCI/Concrete_Data.xls',
                               header=0).values
        elif self.name == "energy":
            data = pd.read_excel(self.data_path+'UCI/ENB2012_data.xlsx',
                                 header=0).values
        elif self.name == "power":
            
            data = pd.read_excel(self.data_path+'UCI/CCPP/Folds5x2_pp.xlsx', header=0).values
        elif self.name == "wine":
            data = pd.read_csv(self.data_path + 'UCI/winequality-red.csv',
                               header=1, delimiter=';').values

        elif self.name == "yacht":
            data = pd.read_csv(self.data_path + 'UCI/yacht_hydrodynamics.data',
                               header=1, delimiter='\s+').values
            
        data = torch.from_numpy(data).float()[:int(subsample * data.shape[0])]
        self.in_dim = data.shape[1] - 1
        self.out_dim = 1
        self.sample_count = data.shape[0]

        if normalize:
            self.std = torch.std(data, dim=0)
            self.mean = torch.mean(data, dim=0)
        else:
            self.std = torch.ones(self.in_dim + self.out_dim)
            self.mean = torch.zeros(self.in_dim + self.out_dim)
        self.data_mean = self.mean[:-1]
        self.data_std = self.std[:-1]
        self.target_mean = self.mean[-1].unsqueeze(-1)
        self.target_std = self.std[-1].unsqueeze(-1)
        data = (data - self.mean) / self.std
        
        self.dataset = torch.utils.data.TensorDataset(data[:,:-1], data[:,-1].unsqueeze(-1))
        test_sample_count = int(test_percentage * self.sample_count)
        self.train_set, self.test_set = torch.utils.data.random_split(self.dataset, [self.sample_count - test_sample_count, test_sample_count])