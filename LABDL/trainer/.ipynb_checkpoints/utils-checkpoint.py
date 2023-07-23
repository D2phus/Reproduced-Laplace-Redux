import matplotlib.pyplot as plt 
import numpy as np

import torch
import torch.nn as nn

def plot(X, y, model): 
        """
        plot the fitting of model on dataset
        """
        preds = model(X).detach().numpy()
        
        fig, axs = plt.subplots()
        fig.tight_layout()
            
        axs.plot(X, y, 'o', label="targeted")
        axs.plot(X, preds, 'o', label="prediction")
        #axs.fill_between(X, 0, preds, color="orange", label="prediction", alpha=0.5) # require 1d input
        axs.legend()
    
       # return fig
    
    