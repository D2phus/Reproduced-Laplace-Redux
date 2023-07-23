import torch 
import torch.nn as nn 

from backpack import backpack, extend
from backpack.extensions import (
    BatchGrad, )


class CurvatureInterface():
    """ Interface to access curvature for a model and corresponding **log likelihood**, which means transition from curvature w.r.t. lossfunc to curvature w.r.t. likelihood is required.
        BackPack is used as the backend to support curvature calculation. 
    """
    def __init__(self, model, regression, stochastic):
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
        jacobians of shape (batch_size, num_params, out_features), where num_params refers to the total number of model parameters
        out of shape (batch_size, out_features): batched model outputs.
        """
        out = self.model(x)
        C = out.shape[-1] # out_features
        
        Js_list = [] 
        for c in range(C): # compute batched derivatives for each dimension output c
            tmp_Js = [] 
            with backpack(BatchGrad(), retain_graph=True): # for full Jacobian, it takes c backpropagations, so retain the graph
                out_c = out[:, c].sum() 
                out_c.backward(retain_graph=True) # store the output in `grad_batch` of shape (batch_size, gradients_size)
            
            for param in self.model.parameters():
                tmp_Js.append(param.grad_batch.detach().flatten(start_dim=1)) # of shape (batch_size, each_param_size), where parameters are flattened to be one-dimensional tensors.
            cat_Js = torch.cat(tmp_Js, dim=1) # of shape (batch_size, num_params)
            Js_list.append(cat_Js)
        Js = torch.stack(Js_list, dim=2) # tensor(batch_size, num_params, out_features)
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
        loss = self.lossfunc(self.model(x), y)
        with backpack(BatchGrad()): 
            loss.backward()# store the output in `grad_batch` of shape (batch_size, gradients_size)
        #gradients = torch.cat([param.grad_batch.detach().flatten(start_dim=1) for param in self.model.parameters()], dim=1) 
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
    
    

