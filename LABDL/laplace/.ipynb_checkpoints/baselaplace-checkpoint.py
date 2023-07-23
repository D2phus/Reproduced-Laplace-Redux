from math import pi, sqrt, log
from torch.nn.utils import parameters_to_vector, vector_to_parameters

import torch
import torch.nn as nn
import torch.distributions as D

import numpy as np

from .curvatures import GGNCurvature, EFCurvature

class BaseLaplace: 
    def __init__(self, 
                 model: nn.Module, 
                 sigma_noise=1.0, 
                 prior_var=1.0, 
                 prior_mean=0.0,  
                 regression: bool=True, 
                 stochastic: bool=False,
                 H_approximation: str='GGN'):
        """The base class for Laplace approximation, extended by FullLaplace / DiagLaplace / KronLaplace
        Args:
        sigma_noise: standard deviation of regression centered Gaussian noise ~ N(0, sigma_noise^2)
        prior_var, scalar: variance of paramter prior, same for each parameter
        prior_mean: mean of parameter prior
        regression: regression task or classification task. 
        stochastic: if stochastic methods are used when computing curvatures.
        H_approximation: ['GGN', 'EF'], the Hessian approximation options: generalized Gauss-Newton, empirical Fisher.
        
        Attrs:
        mean of shape (num_params): mean of approximated Gaussian posterior, i.e. MAP estimate parameters
        backend: curvature backend based on BackPack.
        loss, scalar: empirical training loss, MSELoss for regression and CrossEntropyLoss for classification 
        num_samples, scalar: number of training samples
        num_params, scalar: number of model parameters
        out_features, scalar: output size
        
        
        """
        if H_approximation not in ['GGN', 'EF']: 
            raise ValueError('The Hessian approximation methods have to be GGN (Generalized Gauss-Newton) or EF (Empirical Fisher).')
            
        self.model = model
        self.mean = parameters_to_vector(model.parameters())
        self.num_params = len(self.mean) 
        self.prior_mean = prior_mean
        self.sigma_noise = sigma_noise 
        self.prior_var = torch.tensor(prior_var)
        self.regression = regression
        self.stochastic = stochastic
        
        backend = GGNCurvature if H_approximation == 'GGN' else EFCurvature
        self._backend = None
        self._backend_cls = backend
        
        self.loss = 0 
        self.num_samples = 0
        
    @property
    def backend(self): 
        if self._backend is None:
            self._backend = self._backend_cls(self.model, self.regression, self.stochastic)
        return self._backend
    
    @property
    def prior_mean(self):
        return self._prior_mean
    
    @prior_mean.setter
    def prior_mean(self, prior_mean): 
        """The setter for attribute prior_mean, format the type as torch.Tensor of shape (1) or (num_params)
        Args:
        prior_mean: real scalar, torch.Tensor of shape (1) or (num_params)
        """
        if np.isscalar(prior_mean) and np.isreal(prior_mean):
            self._prior_mean = torch.tensor(prior_mean, dtype=torch.float32)
        elif torch.is_tensor(prior_mean):
            if prior_mean.ndim > 1: 
                raise ValueError('The dimension of prior mean has to be in [0, 1].')
            if len(prior_mean) not in [1, self.num_params]: 
                    raise ValueError('Invalid length of prior mean.')
            self._prior_mean = prior_mean
        else: 
            raise ValueError("Invalid data type for prior mean.")
            
    @property
    def sigma_noise(self):
        return self._sigma_noise
    
    @sigma_noise.setter
    def sigma_noise(self, sigma_noise): 
        """The setter for attribute sigma_noise, format the type as torch.Tensor of shape (1)
        Args:
        prior_mean: real scalar, torch.Tensor of shape (1)
        """
        if np.isscalar(sigma_noise) and np.isreal(sigma_noise):
            self._sigma_noise = torch.tensor(sigma_noise, dtype=torch.float32)
        elif torch.is_tensor(sigma_noise):
            if sigma_noise.ndim > 1: 
                raise ValueError('The dimension of sigma noise has to be in [0, 1].')
            if len(sigma_noise) > 1:  
                    raise ValueError('Invalid length of sigma noise.')
            self._sigma_noise = sigma_noise
        else: 
            raise ValueError("Invalid data type for sigma noise.")
            
    def _init_H(self):
        """init the Hessian matrix"""
        raise NotImplementedError
        
    @property
    def _approximate_H(self, x, y):
        """approximated log-likelihoods Hessian"""
        raise NotImplementedError
        
    @property
    def _H_factor(self):
        """the Hessian factor from lossfunc curvature to likelihood curvature. """
        return 1/(self.sigma_noise**2) # TODO: temperature has not been introduced yet. 
        
    @property
    def prior_precision(self):
        """the prior precision, scalar"""
        return 1/self.prior_var
    
    @property
    def prior_precision_diag(self):
        """the diagonal standard deviation of prior precision of shape (num_params)
        """
        return torch.ones(self.num_params) * self.prior_precision
        
    @property
    def posterior_precision(self):
        """should be called after fit.
        the posterior precision matrix of shape (num_params, num_params)
        \Sigma^{-1} = - \triangledown^2\log h(\theta)|\theta_{MAP} 
        \triangledown^2\log h(\theta)|\theta_{MAP} = -\gamma^{-2}I -\sum_{n=1}^N\triangledown^2_\theta \log p(y_n|f_\theta(x_n))|_{\theta_{MAP} }
        """
        H_prior = torch.diag(self.prior_precision_diag)
        H_lik = self._H_factor * self.H # note to introduce factor
        return H_prior + H_lik
    
    @property
    def posterior_covariance(self): 
        """of shape (num_params, num_params)"""
        return torch.linalg.inv(self.posterior_precision)

    @property
    def log_det_prior_precision(self):
        """the log det of prior precision. """
        return self.prior_precision_diag.log().sum()
        
    @property
    def log_det_posterior_precision(self):
        """the log det of posterior precision"""
        return self.posterior_precision.logdet()    
        
    @property
    def log_det_ratio(self):
        """the log det ratio: log det of prior precision / log det of posterior precision"""
        return self.log_det_prior_precision - self.log_det_posterior_precision 
        
    def fit(self, train_loader, override=True):
        """
        fit the Laplace approximation at the parameters of model. 
        Args:
        train_loader: dataloader of training data.
        override: whether to init H, loss, and num_samples.  
        
        Attrs:
        H: the
        """
        self.model.eval()
        self.mean = parameters_to_vector(self.model.parameters())
        if override: 
            self.loss = 0
            self._init_H()
            self.num_samples = 0
        
        x_sample, y_sample = next(iter(train_loader))
        self.out_features = y_sample.shape[-1] 
        
        for x, y in train_loader: 
            self.model.zero_grad()
            H_batch, loss_batch = self._approximate_H(x, y)
            self.loss += loss_batch
            self.H += H_batch
        
        self.num_samples += len(train_loader.dataset)
        
    @property
    def log_likelihood(self):
        """
        log likelihood p(D|\theta) of all samples, a term of the log marginal likelihood. 
        \sum^N_{n} -\log p(y_n|x_n; \theta) = -N\log \sigma - \frac{N}{2}\log (2\pi) - \frac{MSE_{train}}{2\sigma^2}
        Returns: 
        log_lik, scalar
        """
        factor = -self._H_factor # note the sign
        if self.regression: 
            const = self.out_features * self.num_samples * torch.log(sqrt(2*pi)*self.sigma_noise)
            return factor * self.loss - const
        else: 
            return factor * self.loss
    
    
    def log_marginal_likelihood(self, prior_var=None, sigma_noise=None):
        """should be called after fit. 
        log marginal likelihood: log p(D) for tuning the prior standard deviation and sigma noise. 
        log p(D) =-\sum^N_{n=1}l(x_n, y_n; \theta)+\log p(\theta) + \frac{D}{2}\log(2\pi) + \frac{1}{2} \log(\det \Sigma), note that constant term is ignored when doing optimization. 
        
        Args: 
        prior_var, sigma_noise: hyper-parameters to compute at.
        Returns:
        log_mar_lik, scalar
        """
        # update the hyper-parameter setting
        if prior_var is not None:
            self.prior_var = prior_var 
        if sigma_noise is not None:
            self.sigma_noise = sigma_noise
            
        # const = (self.num_samples - self.num_params)*log(2*pi) # not necesssary for optimization!!!
        diff = self.mean - self.prior_mean 
        scatter = (self.prior_precision_diag*diff)@diff # transpose here is not necessary since diff is one-dimensional. diag is used; for further extension (different prior for different parameters)
        return self.log_likelihood + 1/2*(self.log_det_ratio-scatter)
    
    
    def predict(self, X, pred_type='glm', glm_link_approx='mci', num_samples=100): 
        """The posterior predictive on input X.
        Args: 
        X of shape (batch_size, in_features)
        pred_type: ['glm', 'mci', 'gp'], the posterior predictive approximation options: linearized neural network with gaussian likelihood, MC integration, or gaussian process inference. 
        glm_link_approx: ['mci'], how to approximate the classification link function for the glm.
        
        Returns: 
        mean, variance of the predictive 
        """
        if pred_type == 'mci': 
            samples = self._mci_predictive_samples(X, num_samples)
            if self.regression: 
                return samples.mean(dim=0), samples.var(dim=0)
            else:
                return samples.mean(dim=0), None
            
        elif pred_type == 'glm': 
            f_mu, f_cov = self._glm_output_distribution(X)    
            # + \sigma^2 I?
            if self.regression: 
                return f_mu, f_cov
            else: # classification
                if glm_link_approx == 'mci': 
                    return self.predictive_samples(X, num_samples).mean(dim=0), None
                elif glm_link_approx == 'probit': 
                    pass # TODO
                    
    def predictive_samples(self, X, pred_type='glm', num_samples=100): 
        """samples from the posterior predictive on input X.
        Args: 
        
        Returns: 
        samples of shape (num_samples, batch_size, out_features)
        """
        if pred_type == 'glm': 
            f_mu, f_cov = self._glm_output_distribution(X)
            f_samples = self._normal_samples(f_mu, f_cov, num_samples)
            if not self.regression:
                f_samples = torch.softmax(dim=1)
            return f_samples
        elif self.pred_type == 'mci': 
            return self._mci_predictive_samples(X, num_samples)
    
    def _normal_samples(self, loc, cov, num_samples):
        """sampling from multivariate Gaussian.
        Returns: samples of shape (num_samples, batch_size, out_features)
        """
        samples = D.multivariate_normal.MultivariateNormal(loc, cov).sample([num_samples])
        return samples
        
    def _mci_predictive_samples(self, X, num_samples=100):
        """
        predictive samples for the MC integration. 
        Returns: 
        out of shape (num_samples, batch_size, out_features)
        """
        out = []
        for s in self._normal_samples(self.mean, self.posterior_covariance, num_samples): # parameter samples 
            vector_to_parameters(s, self.model.parameters()) 
            out.append(self.model(X))
        vector_to_parameters(self.mean, self.model.parameters()) # remember to copy back the params.
        out = torch.stack(out, dim=0) 
        if not self.regression: # classification
            out = torch.softmax(out)
        return out
    
    def _glm_output_distribution(self, X):
        """
        The output distribution of a linearized neural network, p(f*|x*, D)\sim N(f*; f_{\theta_{MAP}}, J(x*)^T\Sigma J(x*))
        
        Args: 
        X of shape (batch_size, in_features)
        
        Returns:
        f_mu of shape (batch_size, out_features)
        f_cov of shape (batch_size, out_features, out_features)
        """
        Js, f_mu = self.backend.jacobians(X) 
        f_cov =  torch.einsum("npc, pq, nqk-> nck", Js, self.posterior_covariance, Js)
        return f_mu, f_cov
    