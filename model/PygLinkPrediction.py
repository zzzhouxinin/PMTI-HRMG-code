#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from torch.utils.data.dataset import Dataset
from sklearn import preprocessing

class PygLinkPrediction(Dataset):
    """
    The PygLinkPrediction object can convert graph for pyg using.
    
    Parameters
    ----------
    hete_graph : pd.DataFrame
          --------------------------
         | upper-left | upper-right |
         |------------|-------------|
         | lower-left | lower-right |
          --------------------------
    upper_left_graph : pd.DataFrame
        The upper-left graph.
    lower_right_graph : pd.DataFrame
        The lower-right graph.

    Returns
    -------
    all_node_dict : dict
        key is index, value is node name.
    node1_dict : dict
        key is index in all node, value is node1 name.
    node2_dict : dict
        key is index in all node, value is node2 name.
    all_edges: pd.DataFrame
        With three columns: From, To and Value. 
    """
    
    def __init__(self, 
                 hete_graph: pd.DataFrame,
                 upper_left_graph: pd.DataFrame,
                 lower_right_graph: pd.DataFrame,
                 node_attr: pd.DataFrame):
        
        # Get node name and order
        node1 = upper_left_graph.index.tolist()
        node2 = lower_right_graph.index.tolist()
        all_node = hete_graph.index.tolist()
        
        #hete_graph.loc[node1, node1] = 0
        #hete_graph.loc[node2, node2] = 0
        # Order node attr
        node_attr = node_attr.loc[all_node,]
        
        # Save node name into a dict
        node_name_dict = {v:k for v,k in enumerate(all_node)}
        node1_name_dict = {v:k for v,k in enumerate(node1)}
        node2_name_dict = {v+len(node1):k for v,k in enumerate(node2)}
        
        # Convert node name to num index
        hete_graph = hete_graph.reset_index(drop=True)
        hete_graph.columns = hete_graph.index.tolist()
        
        # Get all edges, three columns COO format
        all_edges = hete_graph.melt(ignore_index=False).reset_index()
        all_edges.columns = ['From','To','Value']
        all_edges['Zvalue'] = preprocessing.scale(all_edges['Value'])
        
        self.all_node_dict = node_name_dict
        self.node1_dict = node1_name_dict
        self.node2_dict = node2_name_dict
        self.all_edges = all_edges
        self.node_attr = node_attr