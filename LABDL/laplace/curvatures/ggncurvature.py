import torch 
import torch.nn as nn 

from .curvatureinterface import CurvatureInterface

from backpack import backpack, extend
from backpack.extensions import (
    DiagGGNExact,
    DiagGGNMC, 
    KFAC, 
    KFLR)

class GGNCurvature(CurvatureInterface): 
    """access curvature for a model and corresponding log likelihood. The Hessian is approximated with generalized Gauss-Newton matrix. """
    def full(self, x, y): 
        """full Hessian approximated with the generalized Gauss-Newton matrix. 
        G:= \sum _n ^N J(x_n) (\triangledown _f ^2 \log p(y_n|f)|_{f=f_{\theta_{MAP}}(x_\theta)})J(x_n)^T
        where $\triangledown _f ^2 \log p(y_n|f)|_{f=f_{\theta_{MAP}}(x_\theta)}$ of shape (out_features, out_features) is a Hessian of log-likelihood of y_n w.r.t. model output. 
        
        1. regression: TODO
        2. classification: turn categorical outputs into probability distribution by applying softmax, 
        then the Hessian is diag(p) - pp^T 
        
        Returns: 
        H_ggn of shape (num_params, num_params)
        loss
        """
        Js, out = self.jacobians(x) # model outputs' Jacobians of shape (batch_size, num_params, out_features), and model output of shape (batch_size, out_features)
        loss = self.lossfunc(out, y)
        
        if self.regression: 
            H_ggn = torch.einsum("imk, ink->mn", Js, Js)
        else: 
            # classification. The 2nd derivative for log likelihood is diag(p) - pp^T
            ps = torch.softmax(out, dim=-1) # turn categorical outputs into probability distribution
            H_lik = torch.diag_embed(ps) - torch.einsum('mk,mc->mck', ps, ps) # of shape (batch_size, out_features, out_features)
            H_ggn = torch.einsum('mpc,mck,mqk->pq', Js, H_lik, Js)
        return H_ggn, self.factor * loss.detach() # note that H_ggn is not derived from log-likelihoods gradients, which means no factor.
    
    def diag(self, x, y):
        """Diagonal of GGN and its MC approximation. 
        
        Args:
        x of shape (batch_size, in_features): batched inputs.
        y of shape (batch_size, out_features): batched targets.

        Returns: 
        H_diag of shape (num_params,): diagonal of the approximated Hessian 
        loss 
        """
        context = DiagGGNMC if self.stochastic else DiagGGNExact
        out = self.model(x)
        loss = self.lossfunc(out, y)
        with backpack(context()):
            loss.backward() # store the output in `diag_ggn_mc`/`diag_ggn_exact` of shape (gradients_shape)
        if self.stochastic: 
            diag_ggn = torch.cat([param.diag_ggn_mc.detach().flatten() for param in self.model.parameters()])
        else: 
            diag_ggn = torch.cat([param.diag_ggn_exact.detach().flatten() for param in self.model.parameters()])
            
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
        #context = KFAC if self.stochastic else KFLR
        #out = self.model(x)
        #loss = self.lossfunc(out, y)
        #with backpack(context()):
        #    loss.backward() # store the output list in `kfac`/`kflr`
            
        # TODO
        #if self.stochastic: 
        #    kron = torch.cat([param.kfac for param in self.model.parameters()])
        #else:
        #    kron = torch.cat([param.kflr for param in self.model.parameters()])
        #return self.factor * kron, self.factor * loss.detach()
    
    