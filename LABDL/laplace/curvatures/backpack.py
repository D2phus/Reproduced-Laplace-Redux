import torch 
import torch.nn as nn 

from .curvature import CurvatureInterface, GGNInterface, EFInterface

from backpack import backpack, extend
from backpack.extensions import (
    BatchGrad, 
    SumGradSquared, 
    DiagGGNExact,
    DiagGGNMC, 
    KFAC, 
    KFLR)


class BackPackInterface(CurvatureInterface): 
    """Curvature inteface based on backpack package."""
    def __init__(self, model, likelihood, last_layer=False, subnetwork_indices=None):
        super().__init__(model, likelihood, last_layer, subnetwork_indices)
        extend(self._model)
        extend(self.lossfunc)
        
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
        # lossfunc = extend(self.lossfunc)
        out = self.model(x)
        loss = self.lossfunc(out, y)
        with backpack(BatchGrad()): 
            loss.backward()
        gradients = torch.cat([param.grad_batch.detach().flatten(start_dim=1) for param in self._model.parameters()], dim=1) 
        return gradients, loss

class BackPackGGN(BackPackInterface, GGNInterface): 
    """backpack-based GGN approximation. """
    def __init__(self, model, likelihood, last_layer=False, subnetwork_indices=None, stochastic=False): 
        super().__init__(model, likelihood, last_layer, subnetwork_indices)
        self.stochastic = stochastic
        
    def diag(self, x, y):
        """Diagonal of GGN and its MC approximation. 
        
        Args:
        x of shape (batch_size, in_features): batched inputs.
        y of shape (batch_size, out_features): batched targets.

        Returns: 
        H_diag of shape (num_params,): diagonal of the approximated Hessian 
        loss 
        """
        context = DiagGGNMC(mc_samples=16) if self.stochastic else DiagGGNExact()
        out = self.model(x)
        loss = self.lossfunc(out, y)
        with backpack(context):
            loss.backward() # store the output in `diag_ggn_mc`/`diag_ggn_exact` of shape (gradients_shape)
        if self.stochastic: 
            diag_ggn = torch.cat([param.diag_ggn_mc.detach().flatten() for param in self._model.parameters()])
        else: 
            diag_ggn = torch.cat([param.diag_ggn_exact.detach().flatten() for param in self._model.parameters()])
        
        if self.subnetwork_indices is not None:
            diag_ggn = diag_ggn[self.subnetwork_indices]
            
        return self.factor * diag_ggn, self.factor * loss.detach()

    def kron(self, x, y):
        """Kronecker Block-Diagonal approximations of the GGN and its MC approximation. 
        
        Args:
        x of shape (batch_size, in_features): batched inputs.
        y of shape (batch_size, out_features): batched targets.
        
        Returns: 
        H_kron of shape (num_kron_factors) 
        loss
        """
        raise ValueError('`kron` method is not available.')
        context = KFAC if self.stochastic else KFLR
        out = self.model(x)
        loss = self.lossfunc(out, y)
        with backpack(context()):
            loss.backward() # store the output list in `kfac`/`kflr`
            

class BackPackEF(BackPackInterface, EFInterface):
    """backpack-based EF approximation. """
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
        
        diag_ef = torch.cat([param.sum_grad_squared.detach().flatten(start_dim=0) for param in self._model.parameters()], dim=0)  # (num_params) 
        
        if self.subnetwork_indices is not None:
            diag_ef = diag_ef[self.subnetwork_indices]
            
        return self.factor * diag_ef, self.factor * loss.detach()
        
    def kron(self, x, y): 
        raise ValueError('`kron` method not available for EF.')
    