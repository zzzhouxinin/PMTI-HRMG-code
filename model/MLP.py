#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from torch import nn

class MLP(torch.nn.Module):
    """
    Multilayer Perceptron.
    
    Nothing fancy, just a simple MLP.
    Default output a one dim embeddings by sigmoid, can be customized.

    Parameters
    ----------
    embedding : torch.tensor
        A tensor containing embeddings from anywhere.
    dim1: int
        Input embeddings dim.
    
    Returns
    -------
    out
        Just a tensor, default only 1 dim.

    """
    
    def __init__(self,
                 dim_input: int,
                 dim_l1: int,
                 dim_l2: int):
        super().__init__()
        self.linear1 = nn.Linear(dim_input,dim_l1)
        self.linear2 = nn.Linear(dim_l1,dim_l2)
        self.linear3 = nn.Linear(dim_l2,1)
    
    def forward(self, embedding):
        """
        Forward propagation.
        
        Default activation functions is sigmoid, can be customized.

        Parameters
        ----------
        embedding : TYPE
            A tensor containing embeddings from anywhere.

        Returns
        -------
        out : TYPE
            DESCRIPTION.

        """
        out = self.linear1(embedding)
        #out = F.relu(out)
        out = self.linear2(out)
        #out = F.relu(out)
        out = self.linear3(out)
        out = torch.sigmoid(out)
        
        return out
