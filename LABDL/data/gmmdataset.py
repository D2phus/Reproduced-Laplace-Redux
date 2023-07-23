import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import DataLoader
from torch.utils.data import random_split 

from typing import Tuple

import matplotlib.pyplot as plt
import scipy.stats as stats

class GMMDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 begin: int, 
                 end: int, 
                 num_samples: int, 
                 num_components=2, 
                 mixture_coefficient=[0.4, 0.6], 
                 means=[-1, 0.5], 
                 stds=[0.4, 0.8], 
                batch_size=64):
        """
        A one-dimensional synthetic example generated via Gaussian mixtures, 
        where each component is a univariate Gaussian distribution 
        
        """
        
        assert sum(mixture_coefficient) == 1
        assert len(means) == num_components
        assert len(stds) == num_components
        self.num_components = num_components
        self.mixture_coefficient = mixture_coefficient
        self.means = means
        self.stds = stds
        self.batch_size = batch_size
        
        X = torch.FloatTensor(num_samples).uniform_(begin, end)
        self.X, _ = torch.sort(X, dim=0)
        
        self.y, self.components = self.gaussian_mixture()
        
        self.setup_dataloaders() 
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        return self.X[idx], self.y[idx]
    
    def plot(self): 
        fig, axs = plt.subplots()
        for index in range(self.num_components): 
            axs.plot(self.X, self.components[index], '-', label=f"component_{index}")
            
        #axs.fill_between(self.X, 0, self.y, color="orange", alpha=0.5)
        axs.plot(self.X, self.y, '-', label="Gaussian mixture")
        axs.legend()
        

    
    def setup_dataloaders(self, 
                          val_split: float = 0.1, 
                          test_split: float = 0.2) -> Tuple[DataLoader, ...]: 
        """
        split the dataset into training set and validation set, and test set
        """
        test_size = int(test_split * len(self))
        val_size = int(val_split * len(self))
        train_size = len(self) - test_size - val_size
        
        train_subset, val_subset, test_subset = random_split(self, [train_size, val_size, test_size])
        
        self.train_dl = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True)
        self.val_dl = DataLoader(val_subset, batch_size=self.batch_size, shuffle=False)
        self.test_dl = DataLoader(test_subset, batch_size=self.batch_size, shuffle=False)
    
    def get_dataloaders(self) -> Tuple[DataLoader, ...]: 
        return self.train_dl, self.val_dl, self.test_dl