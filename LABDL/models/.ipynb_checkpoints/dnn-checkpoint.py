import torch 
import torch.nn as nn
import torch.nn.functional as F 


class DNN(nn.Module):
    def __init__(
        self, 
        in_features: int = 1, 
        out_features: int = 1, 
        dropout_rate: float = 0.0, 
        hidden_sizes: list = [50], 
        activation: bool = True, 
        ) -> None: # type-check
            """
            A simple DNN model for regression tasks. 
            Args:
            in_features: size of each input sample
            out_features: size of each output sample
            dropout_rate: dropout rate for dropout after each hidden layer
            hidden_size: the size of hidden layers. 
            activation: if activation function (Tanh) is used. When it's set as False, the DNN turns into a linear model, which would be more tractable when testing the Bayesian regression. 
            """
            super(DNN, self).__init__()
            dropout = nn.Dropout(p=dropout_rate)
            
            layers = []
            
            layers.append(nn.Linear(in_features, hidden_sizes[0],  bias=True)) # with bias
            #layers.append(nn.ReLU())
            if activation: 
                layers.append(nn.Tanh())
            layers.append(dropout) # dropout 
            
            for in_f, out_f in zip(hidden_sizes[:], hidden_sizes[1:]):
                layers.append(nn.Linear(in_f, out_f,  bias=True))
                if activation: 
                    layers.append(nn.Tanh())
                layers.append(dropout)
                
            layers.append(nn.Linear(hidden_sizes[-1], out_features,  bias=True))
            
            self.model = nn.Sequential(*layers)

    def forward(self, inputs) -> torch.Tensor:
        """
        Args: 
        inputs of shape (batch_size, in_features)
        Returns: 
        outputs of shape (batch_size, out_features)
        """
        return self.model(inputs)
    
    
class LinearSingleLayerModel(nn.Module):
    def __init__(self, in_features: int = 1, out_features: int = 1):
        self.model = nn.Linear(in_features, out_features)
        
    def forward(self):
        model