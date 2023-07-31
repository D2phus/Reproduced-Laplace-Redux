from math import pi, sqrt, log
from torch.nn.utils import parameters_to_vector

import torch
import torch.nn as nn

import numpy as np

from .baselaplace import BaseLaplace


class DiagLaplace(BaseLaplace):
    """
    Laplace approximation using diagonal Hessian approximation.
    """
    _key = ('all', 'diag') 
    
    def __init__(self, 
                 model: nn.Module, 
                 likelihood, 
                 sigma_noise=1.0, 
                 prior_precision=1.0, 
                 prior_mean=0.0,
                 temperature=1.0, 
                 backend=None, 
                 backend_kwargs=None): 
        super().__init__(model, likelihood, sigma_noise, prior_precision, prior_mean, temperature, backend, backend_kwargs)
        if not hasattr(self, 'H'):
            self._init_H()
    
    def _init_H(self): 
        self.H = torch.zeros(self.n_params) # only the diag of Hessian 
        
    def _approximate_H(self, x, y):
        return self.backend.diag(x, y)