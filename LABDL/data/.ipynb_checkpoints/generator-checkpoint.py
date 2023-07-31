"""Dataset class for synthetic data"""
import math 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from typing import Tuple
from typing import Sequence

import matplotlib.pyplot as plt 

def pow_shape(x):
    return 8*torch.pow((x-0.5), 2)

def exp_shape1(x):
    return 0.1 * torch.exp(-8*x+4)

def exp_shape2(x):
    return 5*torch.exp(-2*torch.pow((2*x-1), 2))

def zero_shape(x):
    return torch.zeros_like(x)
    

def get_sin_examples(num_samples=200, 
                    sigma=0.2,
                    batch_size=64):
    """
    generate regression dataset from function \sin(X) with observation centered Gaussian noise ~ N(0, sigma^2)
    
    Args:
    num_samples: the number of training samples
    sigma: the standard deviation of observation noise
    batch_size
    
    Returns: 
    X_train of shape (num_samples): training samples uniformly sampled on interval [0, 8]
    y_train of shape (num_samples): training labels with noise
    train_loader: data loader for the training data
    X_test of shape (500): test samples which are evenly spaced between -5 and 13
    """
    X_train = (torch.rand(num_samples) * 8).unsqueeze(-1)
    y_train = torch.sin(X_train) + torch.randn_like(X_train) * sigma
    trainset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=batch_size
    )
    X_test = torch.linspace(-5, 13, 200).unsqueeze(-1)
    return X_train, y_train, train_loader, X_test

def get_linear_examples(num_samples=200, 
                        slope=0.1,
                        bias=0.2,
                        sigma=0.2, 
                        batch_size=64):
    """
    generate regression dataset from function ax + b with observation centered Gaussian noise ~ N(0, sigma^2)
    
    Args:
    num_samples: the number of training samples
    slope, bias: the slope and bias for the linear function
    sigma: the standard deviation of observation noise
    batch_size
    
    Returns: 
    X_train of shape (num_samples): training samples uniformly sampled on interval [0, 8]
    y_train of shape (num_samples): training labels with noise
    
    train_loader: data loader for the training data
    X_test of shape (500): test samples which are evenly spaced between -5 and 13
    """
    X_train = (torch.rand(num_samples) * 8).unsqueeze(-1)
    y_train = slope*X_train + bias + torch.randn_like(X_train) * sigma
    trainset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=batch_size
    )
    X_test = torch.linspace(-5, 13, 500).unsqueeze(-1)
    return X_train, y_train, train_loader, X_test


class ToyDataset(torch.utils.data.Dataset):
    def __init__(self,
                 task_name: str, 
                 config, 
                 num_samples: int, 
                 in_features: int, 
                 x_start: float, 
                 x_end: float, 
                 gen_funcs: Sequence,
                 gen_func_names: Sequence, 
                 use_test: bool = False, 
                 )-> None:
        """
        dataset generated with additive model consisted of synthetic functions. 
        
        Args:
        task_name: indicates the name for this task
        num_samples: the number of samples
        [x_start, x_end]: the x-value range for sampling. X is sampled uniformly.
        in_features: the size of input samples
        gen_funcs: list of synthetic functions for input features
        gen_func_names: list of synthetic function names
        
        Property: 
        X of shape (batch_size, in_features)
        y of shape (batch_size)
        feature_outs of shape (batch_size, in_features)
        """
        super(ToyDataset, self).__init__()
        self.task_name = task_name
        self.config = config 
        self.num_samples = num_samples
        self.in_features = in_features
        self.gen_funcs = gen_funcs
        self.gen_func_names = gen_func_names
        
        # uniformly sampled X
        self.X = torch.FloatTensor(num_samples, in_features).uniform_(x_start, x_end)
#         X = ((x_end - x_start) * torch.rand(num_samples, in_features) + x_start).type(torch.FloatTensor)
        if use_test: 
            self.X, _ = torch.sort(self.X, dim=0) # don't sort when training => correlation! 
        
        self.feature_outs = torch.stack([gen_funcs[index](x_i) for index, x_i in enumerate(torch.unbind(self.X, dim=1))], dim=1) # (batch_size, in_features) 
        
        y = self.feature_outs.sum(dim=1) # of shape (batch_size)
        self.y = y + gaussian_noise(y) # y = f(x) + e, where e is random Gaussian noise generated from N(0, 1)
        
        self.setup_dataloaders() 
       
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        return self.X[idx], self.y[idx]

    def plot(self):
        """
        plot each features on the whole dataset.
        """
        cols = 4
        rows = math.ceil(self.in_features / cols)
        fig, axs = plt.subplots(rows, cols)
        fig.tight_layout()
        for index in range(self.in_features): 
            col = index % cols 
            row = math.floor(index / cols)
            axs[row, col].plot(self.X[:, index], self.feature_outs[:, index], '.')
            axs[row, col].set_title(f"X{index}")
#         axs[-1].plot(self.X[:, 0], self.y, '.')
#         axs[-1].set_title(self.task_name)
        
    def plot_subset(self):
        """
        plot training, validation, and test subset 
        """
        subsets = [self.train_subset, self.val_subset, self.test_subset]
        subset_names = ['training set', 'validation set', 'test set']
        for row_index, subset in enumerate(subsets): 
            indices = sorted(subset.indices)
            fig, axs = plt.subplots(1, self.in_features+1, figsize=(10, 2), constrained_layout=True) 
            fig.suptitle(f"{subset_names[row_index]}")
            
            for index in range(self.in_features): 
                axs[index].plot(self.X[indices, index], self.feature_outs[indices, index], '.')
                axs[index].set_title(self.gen_func_names[index])
            axs[-1].plot(self.X[indices, 0], self.y[indices], '.')
            axs[-1].set_title(self.task_name)

        
    def setup_dataloaders(self, val_split: float = 0.1, test_split: float = 0.2) -> Tuple[DataLoader, ...]: 
        """
        split the dataset into training set and validation set, and test set
        """
        test_size = int(test_split * len(self))
        val_size = int(val_split * len(self))
        train_size = len(self) - test_size - val_size
        
        train_subset, val_subset, test_subset = random_split(self, [train_size, val_size, test_size])
        
        self.train_subset, self.val_subset, self.test_subset = train_subset, val_subset, test_subset
        
        self.train_dl = DataLoader(train_subset, batch_size=self.config.batch_size, shuffle=True)
        self.val_dl = DataLoader(val_subset, batch_size=self.config.batch_size, shuffle=False)
        self.test_dl = DataLoader(test_subset, batch_size=self.config.batch_size, shuffle=False)
    
    def get_dataloaders(self) -> Tuple[DataLoader, ...]: 
        return self.train_dl, self.val_dl, self.test_dl
    