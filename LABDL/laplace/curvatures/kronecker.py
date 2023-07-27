import torch
import torch.nn as nn

class Kron: 
    def __init__(self, kronB):
        self.kronB = kronB
        
    def