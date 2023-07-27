import torch 
import torch.nn as nn 

from backpack import backpack, extend
from backpack.extensions import (
    BatchGrad, )


class CurvatureInterface():
    """ Interface to access curvature for a model and corresponding **log likelihood**, which means transition from curvature w.r.t. lossfunc to curvature w.r.t. likelihood is required.
        BackPack is used as the backend to support curvature calculation. 
    """
    def __init__(self, model, regression=True, stochastic=False):
        """
        Args:
        regression: bool, solving regression tasks or classification tasks. 
        stochastic: bool, using exact computation or MC approximation. 
        
        Attrs:
        lossfunc: function for accumulated loss, MSELoss for regression tasks and CrossEntropyLoss for classification tasks. 
        factor: the const factor for transition from lossfunc curvature to log likelihood curvature. 1/2 for regression tasks and 1 for classification tasks. 
        \sum^N_{n} \log p(y_n|x_n; \theta) = -N \cdot C\log \sigma - \frac{N \cdot C}{2}\log (2\pi) - \frac{MSE_{train}}{2\sigma^2}
        """
        self.model = extend(model) # track the computational graph with BackPack
        self.regression = regression 
        lossfunc = nn.MSELoss(reduction='sum') if regression else nn.CrossEntropyLoss(reduction='sum')
        self.lossfunc = extend(lossfunc) # track the computational graph with BackPack
        self.factor = 0.5 if regression else 1.0 
        self.stochastic = stochastic 
    
    def jacobians(self, x: torch.Tensor):
        """Batched Jacobians at current parameters w.r.t. model output (per dimension)
        Args: 
        x of shape (batch_size, in_features): batched inputs.
        
        Returns: 
        jacobians of shape (batch_size, out_features, num_params). parameters are flatten into 1-dimensional
        out of shape (batch_size, out_features): batched model outputs.
        """
        model = extend(self.model)
        Js_list = [] 
        for c in range(model.out_features): # compute batched derivatives for each dimension output c
            out = model(x) # setting retain_graph as True is expensive; forward everytime instead 
            tmp_Js = [] 
            with backpack(BatchGrad()): 
                out_c = out[:, c].sum() 
                out_c.backward() # store the output in `grad_batch` of shape (batch_size, gradients_size)
            
            for param in model.parameters():
                tmp_Js.append(param.grad_batch.detach().flatten(start_dim=1)) 
            cat_Js = torch.cat(tmp_Js, dim=1) # of shape (batch_size, num_params)
            Js_list.append(cat_Js)
        Js = torch.stack(Js_list, dim=2).transpose(1, 2)
        return Js, out
    
    def gradients(self, x, y):
        """Batched gradients at current parameters w.r.t. loss function. 
        note that to obtain gradients w.r.t. the log likelihood, factor should be applied. 
        
        Args: 
        x of shape (batch_size, in_features): batched inputs.
        y of shape (batch_size, out_features): batched targets.
        
        Returns:
        gradients of shape (batch_size, num_params) 
        loss
        """
        lossfunc = extend(self.lossfunc)
        out = self.model(x)
        loss = lossfunc(out, y)
        with backpack(BatchGrad()): 
            loss.backward()
        gradients = torch.cat([param.grad_batch.detach().flatten(start_dim=1) for param in self.model.parameters()], dim=1) 
        return gradients, loss
    
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
    
    

