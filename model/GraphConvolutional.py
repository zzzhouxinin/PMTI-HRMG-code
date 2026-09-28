#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, LayerNorm

class GraphConvolutional(torch.nn.Module):
    """
    Graph convolutional network.
    
    Default three layers and output n dim embeddings, can be customized.
    
    Hidden dimensions can also be customized.
    
    Parameters
    ----------
    node_attr : torch.tensor
        A tensor containing node attributes.
    dim1: int
        The dim of layer 1 output.
    dim2: int
        The dim of layer 2 output.
    dim3: int
        The dim of layer 3 (final layer) output.
        
    Returns
    -------
    h
        GCN embeddings.

    """
    
    def __init__(self,
                 node_attr: torch.tensor,
                 dim_l1: int,
                 dim_l2: int,
                 dim_output: int):
        
        super().__init__()
        self.conv1 = GCNConv(node_attr.shape[1], dim_l1, improved=False, cached=True)
        self.conv2 = GCNConv(dim_l1, dim_l2, improved=False, cached=True)
        self.conv3 = GCNConv(dim_l2, dim_output, improved=False, cached=True)
        
    def forward(self, x: torch.tensor, edge_index: torch.tensor, edge_weight: torch.tensor):
        """
        Forward propagation.
        
        Default activation functions is relu, can be customized.
        
        Parameters
        ----------
        x : torch.tensor
            A tensor containing node attributes, same as node_attr.
        edge_index : torch.tensor
            A tensor containing edges.
        edge_weight: torch.tensor
            A tensor containing edge weight.

        Returns
        -------
        h : TYPE
            GCN embeddings.

        """
        h = self.conv1(x, edge_index, edge_weight)
        h = F.celu(h)
        h = self.conv2(h, edge_index, edge_weight)
        h = F.celu(h)
        h = self.conv3(h, edge_index, edge_weight)
        h = F.celu(h)  # Final GNN embedding space.
    
        return h