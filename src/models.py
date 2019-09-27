import os
import torch
import torch.nn as nn
import numpy as np

class LinearModel(nn.Module):
    def __init__(self, F, T, h = 300):
        super(LinearModel, self).__init__()
        self.M = torch.nn.Linear(F*T, h)
    
    def forward(self, x):
        res = self.M(x.view(x.shape[0], -1))
        return res