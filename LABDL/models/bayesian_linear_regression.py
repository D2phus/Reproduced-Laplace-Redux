import torch 
from math import pi, log

class BayesianLinearRegression(): 
    """Bayesian Linear Regression with Gaussian prior and likelihood."""
    def __init__(self, X, y, sigma_noise=1.0, prior_var=1.0):
        """
        Linear regression: \hat{y} = \omega^T x + \epsilon.
        X of shape (num_samples, num_params-1), y of shape (num_samples, out_features): the training samples
        sigma_noise, scalar: the standard deviation of centered Gaussian noise
        prior_var, scalar: the variance of parameter prior, same for all parameters. 
        
        Attrs: 
        X of shape (num_samples, num-params)
        num_samples, scalar: the number of training samples
        num_params, scalar: the number of parameters
        """
        self.X = torch.cat((X, torch.ones(X.shape[0], 1)), 1) # add a all-ones feature dimension to X for bias
        self.y = y
        self.sigma_noise = torch.tensor(sigma_noise)
        self.prior_var = torch.tensor(prior_var)
        
        self.num_samples, self.num_params = self.X.shape
        
    @property 
    def noise_precision(self): 
        return 1/(self.sigma_noise**2)
    
    @property
    def prior_precision(self):
        return 1/self.prior_var
    
    @property
    def prior_precision_diag(self):
        """the diagonal of prior precision of shape (num_params)"""
        return torch.ones(self.num_params)*self.prior_precision
    
    @property 
    def posterior_precision(self): 
        """the parameter posterior precision of shape (num_params, num_params)"""
        cov_lik = self.noise_precision*(self.X.T@self.X)
        cov_prior = torch.diag(self.prior_precision_diag)
        return cov_lik + cov_prior
    
    @property
    def posterior_var(self): 
        """the parameter variance of shape (num_params, num_params)"""
        return torch.linalg.inv(self.posterior_precision)
        
    @property 
    def mean(self): 
        """the parameter posterior mean of shape (num_params, 1)"""
        return (self.noise_precision*self.posterior_var @self.X.T@self.y)
    
    @property
    def log_det_posterior_precision(self): 
        return self.posterior_precision.logdet()
    
    def log_marginal_likelihood(self, prior_var=None, sigma_noise=None):
        """the logarithm of marignal likelihood, i.e. evidence, p(D) """
        
        # M: num_params, N: num_samples, mN: mean, SN: variance, alphas: prior_precision, beta: noise_precision
        if prior_var is not None: 
            self.prior_var = prior_var
        if sigma_noise is not None: 
            self.sigma_noise = sigma_noise
        
        diff = self.y - self.X@self.mean
        scatter_lik = self.noise_precision*torch.sum(diff**2)
        scatter_prior= self.prior_precision*torch.sum(self.mean**2)
        scatter = scatter_lik + scatter_prior
        
        # note that to compute gradients of parameters, torch.log which has grad attribute should be used, instead of math.log which returns float type!!!
        log_marg_lik = self.num_params*torch.log(self.prior_precision)+self.num_samples*torch.log(self.noise_precision)-scatter-self.log_det_posterior_precision-self.num_samples*log(2*pi) 
        return 0.5*log_marg_lik
    
    def predict(self, X_test): 
        """the prediction p(y|\omega, D)
        Args: 
        X_test of shape (num_samples, num_params-1)
        Returns: 
        pred_mean of shape (num_samples)
        pred_var of shape (num_samples, num_samples)
        """
        num_samples = X_test.shape[0]
        X_test = torch.cat((X_test, torch.ones(X_test.shape[0], 1)), 1) 
        pred_mean = X_test@self.mean
        pred_var = torch.diag(torch.ones(num_samples)*(self.sigma_noise**2)) + X_test@self.posterior_var@X_test.T
        return pred_mean, pred_var
        
    