import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.nn.utils import parameters_to_vector

from copy import deepcopy

import numpy as np
from .curvatures import BackPackGGN
from .laplace import Laplace

def marglik_training(model,
                     train_loader,
                     likelihood, 
                     
                     backend=BackPackGGN,
                     hessian_structure='full',
                     
                     optimizer_cls=torch.optim.Adam, 
                     optimizer_kwargs=None, 
                     scheduler_cls=None,
                     scheduler_kwargs=None,
                     
                     n_epochs = 300,
                     lr_hyp = 1e-1,
                     prior_structure='scalar', 
                     n_epochs_burnin=0, 
                     n_hypersteps=10, 
                     marglik_frequency = 1,
                     
                     prior_prec_init=1.0, 
                     sigma_noise_init=1.0, 
                     temperature=1.0,
                     ): 
    """
    online learning the hyper-parameters.
    the prior p(\theta_i)=N(\theta_i; 0, \gamma^2)
    Args:
    prior_structure: ['scalar', 'layerwise', 'diagonal'], corresponding to the same prior for all parameters / same prior for each weight / different prior for each parameter.
    
    """    
    N = len(train_loader.dataset)
    H = len(list(model.parameters()))
    P = len(parameters_to_vector(model.parameters()))
    # set up hyperparameters and loss function
    hyperparameters = list()
    log_prior_prec_init = np.log(temperature*prior_prec_init)
    if prior_structure == 'scalar':
        log_prior_prec = log_prior_prec_init * torch.ones(1)
    elif prior_structure == 'layerwise':
        log_prior_prec = log_prior_prec_init * torch.ones(H)
    elif prior_structure == 'diagonal':
        log_prior_prec = log_prior_prec_init * torch.ones(P)
    else:
        raise ValueError('invalid prior structure.')
    log_prior_prec.requires_grad = True # note to require grad
    hyperparameters.append(log_prior_prec)

    if likelihood == 'regression': 
        criterion = nn.MSELoss(reduction='mean')
        log_sigma_noise_init = np.log(sigma_noise_init)
        log_sigma_noise = torch.ones(1)*log_sigma_noise_init
        log_sigma_noise.requires_grad = True
        hyperparameters.append(log_sigma_noise)
    elif likelihood == 'classification':
        criterion = nn.CrossEntropyLoss(reduction='mean')
        sigma_noise = 1.
    else: 
        raise ValueError('likelihood should be `regression` or `classification`. ')
        
    # set up model optimizer
    if optimizer_kwargs is None:
        optimizer_kwargs = dict()
    optimizer = optimizer_cls(model.parameters(), **optimizer_kwargs)
    
    # set up scheduler
    if scheduler_cls is not None:
        if scheduler_kwargs is None:
            scheduler_kwargs = dict()
        scheduler = torch.optim.scheduler_cls(optimizer, **scheduler_kwargs)
    
    # set up hyperparameter optimizer
    hyper_optimizer = torch.optim.Adam(hyperparameters, lr=lr_hyp)
    
    best_marglik = np.inf
    best_model_dict = None
    best_precision = None
    best_sigma = None
    margliks = list()
    losses = list()
    for epoch in range(n_epochs):
        epochs_loss = 0.0
        epoch_perf = 0
        for X, y in train_loader:
            optimizer.zero_grad()
            if likelihood == 'regression':
                sigma_noise = log_sigma_noise.exp().detach()
                crit_factor = temperature / (2 * sigma_noise.square())
            else:
                crit_factor = temperature
            prior_prec = log_prior_prec.exp().detach()
            theta = parameters_to_vector(model.parameters())
            delta = expand_prior_precision(prior_prec, model)
            f = model(X)
            # step_loss = crit_factor*criterion(f, y) + (0.5 * (delta * theta) @ theta) / N 
            step_loss = criterion(f, y) + (0.5 * (delta * theta) @ theta) / N / crit_factor
            step_loss.backward()
            optimizer.step()
                
            epochs_loss += step_loss*len(y)
            if likelihood == 'regression': 
                epoch_perf += (f.detach() - y).square().sum()
            else:
                epoch_perf += torch.sum(torch.argmax(f.detach(), dim=-1) == y).item()
            if scheduler_cls is not None:
                scheduler.step()
            
        losses.append(epochs_loss / N)   
        # optimize hyper-parameters
        if epoch >= n_epochs_burnin and epoch % marglik_frequency == 0:
            # fit laplace approximation 
            sigma_noise = 1 if likelihood == 'classification' else log_sigma_noise.exp()
            prior_prec = log_prior_prec.exp()
            la = Laplace(model=model, likelihood=likelihood, hessian_structure=hessian_structure, 
                        subset_of_weights='all', backend=backend, sigma_noise=sigma_noise, prior_precision=prior_prec, temperature=temperature)
            la.fit(train_loader)
            
            # maximize the marginal likelihood
            for _ in range(n_hypersteps):
                hyper_optimizer.zero_grad()
                if likelihood == 'classification': # sigma_noise will be constant 1 for classification. 
                    sigma_noise = None 
                else:
                    sigma_noise = log_sigma_noise.exp()
                    
                prior_prec = log_prior_prec.exp()
                neg_log_marglik = -la.log_marginal_likelihood(prior_prec, sigma_noise)
                neg_log_marglik.backward()
                hyper_optimizer.step()
                margliks.append(neg_log_marglik.item())
            
            # best model selection.
            if margliks[-1] < best_marglik:
                best_model_dict = deepcopy(model.state_dict())
                best_precision = deepcopy(prior_prec.detach())
                best_sigma = 1 if likelihood == 'classification' else deepcopy(sigma_noise.detach())
                best_marglik = margliks[-1]
            
    print('MARGLIK: finished training. Recover best model and fit Laplace.')
    if best_model_dict is not None: 
        model.load_state_dict(best_model_dict)
        sigma_noise = best_sigma
        prior_prec = best_precision
    la = Laplace(model=model, likelihood=likelihood, hessian_structure=hessian_structure, 
                        subset_of_weights='all', backend=backend, sigma_noise=sigma_noise, prior_precision=prior_prec, temperature=temperature)
    la.fit(train_loader)
    return la, model, margliks, losses
    

def expand_prior_precision(prior_prec, model):
    """expand the prior precision variable to shape (num_params)
    Args: 
    H: the number of layers
    P: the number of parameters
    """
    assert prior_prec.ndim == 1 
    params = list(model.parameters())
    H = len(params) # number of layers
    P = len(parameters_to_vector(model.parameters())) # number of parameters
    
    prec_len = len(prior_prec)
    if prec_len == 1:
        return torch.ones(P)*prior_prec
    elif prec_len == H:
        torch.cat([torch.flatten(torch.ones_like(param) * prec) for prec, param in zip(prior_prec, params)], dim=0)
    elif prec_len == P:
        return prior_prec
    else: 
        raise ValueError('invalid shape of prior precision.')