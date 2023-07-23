from math import pi, sqrt, log
from torch.nn.utils import parameters_to_vector

import torch
import torch.nn as nn

import numpy as np

from .curvatures import GGNCurvature, EFCurvature
from .baselaplace import BaseLaplace

class FullLaplace(BaseLaplace): 
    """
    Laplace approximation using full Hessian approximation.
    """
    def __init__(self, 
                 model, 
                 sigma_noise=1.0, 
                 prior_var=1.0, 
                 prior_mean=0.0,  
                 regression=True, 
                 stochastic=False, 
                 H_approximation='GGN'): 
        super().__init__(model, sigma_noise, prior_var, prior_mean, regression, stochastic, H_approximation)
        if not hasattr(self, 'H'):
            self._init_H()
            
    def _init_H(self): 
        self.H = torch.zeros(self.num_params, self.num_params)
        
    def _approximate_H(self, x, y):
        return self.backend.full(x, y)