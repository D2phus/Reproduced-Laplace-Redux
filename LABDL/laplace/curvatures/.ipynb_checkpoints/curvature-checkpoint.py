import torch 
import torch.nn as nn 


class CurvatureInterface:
    """ Interface to access curvature for a model and corresponding **log likelihood**, which means transition from curvature w.r.t. lossfunc to curvature w.r.t. likelihood is required.
        BackPack is used as the backend to support curvature calculation. 
    """
    def __init__(self, model, likelihood, last_layer=False, subnetwork_indices=None):
        """
        Args:
        likelihood: regression tasks or classification tasks. 
        Attrs:
        lossfunc: function for accumulated loss, MSELoss for regression tasks and CrossEntropyLoss for classification tasks. 
        factor: the const factor for transition from lossfunc curvature to log likelihood curvature. 1/2 for regression tasks and 1 for classification tasks. 
        \sum^N_{n} \log p(y_n|x_n; \theta) = -N \cdot C\log \sigma - \frac{N \cdot C}{2}\log (2\pi) - \frac{MSE_{train}}{2\sigma^2}
        """
        if likelihood not in ['classification', 'regression']:
            raise ValueError('likelihood should be `classification` or `regression`.')
            
        self.likelihood = likelihood
        self.model = model 
        self.lossfunc = nn.MSELoss(reduction='sum') if likelihood == 'regression' else nn.CrossEntropyLoss(reduction='sum')
        self.factor = 0.5 if likelihood == 'regression' else 1.0 
        self.last_layer = last_layer
        self.subnetwork_indices = subnetwork_indices
        
    @property
    def _model(self):
        return self.model.last_layer if self.last_layer else self.model
        
    def jacobians(self, x: torch.Tensor):
        raise NotImplementedError
        
    def gradients(self, x, y):
        raise NotImplementedError
        
    def full(self, x, y): 
        """full Hessian approximation. can be implemented with:
        1. the empirical Fisher by sampling the training targets instead of taking the expectation.
        2. the generalized Gauss-Newton (GGN) matrix, which is equivalent to the standard Fisher on classification tasks. 
        
        
        Args:
        x of shape (batch_size, in_features): batched inputs.
        y of shape (batch_size, out_features): batched targets.
        
        Returns: 
        H of shape (num_params, num_params)
        loss detached from the computational graph
        """
        raise NotImplementedError
        
    def diag(self, x, y):
        """Diagonal factorized Hessian approximation. 
        implemented by taking the diagonal of GGN and its MC approximation. 
        
        Args:
        x of shape (batch_size, in_features): batched inputs.
        y of shape (batch_size, out_features): batched targets.
        
        Returns: 
        H_diag of shape (num_params): the diagonal of approximated Hessian. 
        loss 
        """
        raise NotImplementedError
        
    def kron(self, x, y):
        """Kronecker Block-Diagonal approximations of the GGN and its MC approximation. 
        
        Args:
        x of shape (batch_size, in_features): batched inputs.
        y of shape (batch_size, out_features): batched targets.
        
        Returns: 
        H_kron of shape (num_kron_factors) 
        loss
        """
        raise NotImplementedError


class GGNInterface(CurvatureInterface):
    """Curvature interface with Generalized Gauss-Newton approximation. 
    Args: 
    stochastic: using exact method or MC method.
    """
    def __init__(self, model, likelihood, last_layer=False, subnetwork_indices=None, stochastic=False):
        self.stochastic = stochastic
        super().__init__(model, likelihood, last_layer, subnetwork_indices)
        
    def full(self, x, y): 
        """full Hessian approximation.
        G:= \sum _n ^N J(x_n) (\triangledown _f ^2 \log p(y_n|f)|_{f=f_{\theta_{MAP}}(x_\theta)})J(x_n)^T
        where $\triangledown _f ^2 \log p(y_n|f)|_{f=f_{\theta_{MAP}}(x_\theta)}$ of shape (out_features, out_features) is a Hessian of log-likelihood of y_n w.r.t. model output. 
        
        1. regression: Identity.
        2. classification: Softmax.
        the Hessian of Softmax: diag(p) - pp^T.
        
        Returns: 
        H_ggn of shape (num_params, num_params)
        loss
        """
        if self.stochastic: 
            raise ValueError('stochastic method not available for full GGN. ')
            
        Js, out = self.jacobians(x) # model outputs' Jacobians of shape (batch_size, out_features, num_params), and model output of shape (batch_size, out_features)
        loss = self.lossfunc(out, y)
        
        if self.likelihood == 'regression': 
            H_ggn = torch.einsum("ikm, ikn->mn", Js, Js)
        else: 
            # classification
            p = torch.softmax(out, dim=-1) 
            H_lik = torch.diag_embed(p) - torch.einsum('mk,mc->mck', p, p) # of (batch_size, out_features, out_features)
            H_ggn = torch.einsum('mcp,mck,mkq->pq', Js, H_lik, Js)
        return H_ggn, self.factor * loss.detach() # note that H_ggn is not derived from log-likelihoods gradients, which means no factor.
    
    
class EFInterface(CurvatureInterface): 
    """Curvature interface with empirical Fisher approximation, which samples the training targets instead of taking the expectation. """
    def full(self, x, y): 
        """full Hessian approximation. 
        Args:
        x of shape (batch_size, in_features): batched inputs.
        y of shape (batch_size, out_features): batched targets.
        
        Returns: 
        H_full of shape (num_params, num_params)
        loss detached from the computational graph
        """
        Gs, loss = self.gradients(x, y) # Gs of shape (batch_size, num_params)
        H_ef = Gs.T@Gs # of shape (num_params, num_params) = sum up the batch
        return self.factor*H_ef, self.factor*loss.detach() 
        