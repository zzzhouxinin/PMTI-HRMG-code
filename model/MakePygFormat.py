#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
from torch.utils.data.dataset import Dataset
from sklearn import preprocessing

class MakePygFormat(Dataset):
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
                 upper_left: pd.DataFrame,
                 upper_right: pd.DataFrame,
                 lower_left: pd.DataFrame,
                 lower_right: pd.DataFrame,
                 node_attr: pd.DataFrame):
        
        # Construct heterogeneous graph
        upper = pd.concat([upper_left, upper_right], axis=1)
        lower = pd.concat([lower_left, lower_right], axis=1)
        hete = pd.concat([upper, lower], axis=0)
        attr = node_attr.loc[hete.index,]
        
        # Get node name and order
        node1 = upper_left.index.tolist()
        node2 = lower_right.index.tolist()
        all_node = hete.index.tolist()
        
        #hete_graph.loc[node1, node1] = 0
        #hete_graph.loc[node2, node2] = 0
        
        # Save node name into a dict
        node_index_name_dict = {v:k for v,k in enumerate(all_node)}
        node_name_index_dict = {k:v for v,k in enumerate(all_node)}
        
        node1_index_name_dict = {v:k for v,k in enumerate(node1)}
        node1_name_index_dict = {k:v for v,k in enumerate(node1)}
        node2_index_name_dict = {v+len(node1):k for v,k in enumerate(node2)}
        node2_name_index_dict = {k:v+len(node1) for v,k in enumerate(node2)}
        
        # Convert node name to num index
        hete = hete.reset_index(drop=True)
        hete.columns = hete.index.tolist()
        
        # Get all edges, three columns COO format
        all_edges = hete.melt(ignore_index=False).reset_index()
        all_edges.columns = ['From','To','Value']
        all_edges['Zvalue'] = preprocessing.scale(all_edges['Value'])
        
        self.all_node_index_name_dict = node_index_name_dict
        self.all_node_name_index_dict = node_name_index_dict
        self.node1_index_name_dict = node1_index_name_dict
        self.node1_name_index_dict = node1_name_index_dict
        self.node2_index_name_dict = node2_index_name_dict
        self.node2_name_index_dict = node2_name_index_dict
        self.all_edges = all_edges
        self.node_attr = attr