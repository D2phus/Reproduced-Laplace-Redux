import torch 
import torch.nn as nn 

from .curvatureinterface import CurvatureInterface

from backpack import backpack, extend
from backpack.extensions import (
    SumGradSquared, )

class EFCurvature(CurvatureInterface): 
    """access curvature for a model and corresponding log likelihood. The Hessian is approximated with the emprical Fisher. """
    def full(self, x, y): 
        """full Hessian approximated with the empirical Fisher by sampling the training targets instead of taking the expectation.
        
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
        
    def diag(self, x, y): 
        """Diagonal of the empirical Fisher, i.e. for each parameter, it is the sum of individual-squared gradients in the batch.
        
        Args:
        x of shape (batch_size, in_features): batched inputs.
        y of shape (batch_size, out_features): batched targets.

        Returns: 
        H_diag of shape (num_params,): diagonal of the approximated Hessian 
        loss 
        """
        loss = self.lossfunc(self.model(x), y)
        with backpack(SumGradSquared()):
            loss.backward() # store the output in `sum_grad_squared` of shape (gradients_size)
        
        diag_ef = torch.cat([param.sum_grad_squared.detach().flatten(start_dim=0) for param in self.model.parameters()], dim=0)  # (num_params) 
        return self.factor * diag_ef, self.factor * loss.detach()
        
    def kron(self, x, y): 
        raise ValueError('`kron` method is not available.')
    