#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import (
    GCNConv,
    ChebConv,
    JumpingKnowledge,
)

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim, activate_last=False):
        super().__init__()
        dims = [in_dim] + hidden_dims + [out_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 1:
                layers.append(nn.LayerNorm(dims[i+1]))
                # use CELU consistent with your baseline
                layers.append(nn.CELU(inplace=True))
        if not activate_last:
            # remove last activation & norm
            layers = layers[:-2]  # remove last LayerNorm & CELU
            layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class HRMG(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dims,
        out_dim,
        cheb_Ks=(2,4,6), # K values for multi-scale Cheb branches
        use_jk=False,
        dropout=0.2,
    ):
        super().__init__()
        self.use_jk = use_jk
        self.dropout = dropout

        # Input projection
        proj_dim = hidden_dims[0]
        self.input_proj = nn.Linear(in_dim, proj_dim)  # Feature semantic alignment + graph modeling adaptation layer

        # Layer 1: Parallel multi-scale Cheb branches
        self.cheb_branches_l1 = nn.ModuleList()
        for K in cheb_Ks:
            self.cheb_branches_l1.append(ChebConv(proj_dim, hidden_dims[0], K=K))

        # A separate LayerNorm for each parallel multi-scale Chebyshev spectral convolution branch,
        # used for feature normalization and stabilization of branch outputs.
        self.norms_l1 = nn.ModuleList([nn.LayerNorm(hidden_dims[0]) for _ in range(len(cheb_Ks))])

        # Multi-scale attention (automatically determines which scale is more important for each node)
        self.scale_attn = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[0] // 2),  # Reduce dimension to save parameters
            nn.CELU(inplace=True),  # Enable non-linear expressiveness for the attention MLP
            nn.Linear(hidden_dims[0] // 2, 1)  # Output one attention weight per branch
        )

        # Layer 2: GCN (takes attention-weighted fused features as input)
        self.gcn_l2 = GCNConv(hidden_dims[0], hidden_dims[1], cached=False)

        # Normalization layers
        self.norm_l2 = nn.LayerNorm(hidden_dims[1])
        self.norm_l3 = nn.LayerNorm(out_dim)

        # JK module (a cross-layer feature fusion mechanism)
        if use_jk:
            self.jk = JumpingKnowledge(mode='cat')  # Concatenate, then project back to out_dim
            jk_input_dim = hidden_dims[0] + hidden_dims[1]
            self.jk_proj = nn.Linear(jk_input_dim, out_dim)

    def forward(self, x, edge_index, edge_weight):
        # Input projection layer
        h0 = self.input_proj(x)     # [N, 128]
        h0 = F.celu(h0)

        # ========== Layer 1: Multi-branch spectral graph convolution ==========
        branch_outputs = []
        attn_scores = []
        idx = 0
        for i, cheb in enumerate(self.cheb_branches_l1):
            hi = cheb(h0, edge_index, edge_weight)
            hi = F.celu(hi)
            hi = self.norms_l1[idx](hi); idx += 1
            branch_outputs.append(hi)
            attn_scores.append(self.scale_attn(hi))   # [N, 1]

        # [N, S, 1]
        attn_scores = torch.stack(attn_scores, dim=1)
        attn_weights = F.softmax(attn_scores, dim=1) # Get attention weights per node across scales, i.e. which branch the node relies more on

        # Stack branch node features into a 3D tensor for attention-weighted fusion
        branch_outputs = torch.stack(branch_outputs, dim=1)

        # ----- Attention fusion -----
        h_af = torch.sum(attn_weights * branch_outputs, dim=1) # Weighted sum of multi-scale features using attention weights
        h_af = F.dropout(h_af, p=self.dropout, training=self.training) # Prevent over-reliance on a single scale, improve generalization

        # dropout
        h_af = F.dropout(h_af, p=self.dropout, training=self.training)

        # ========== Layer 2 ==========
        h2 = self.gcn_l2(h_af, edge_index, edge_weight)
        h2 = F.celu(h2)
        h2 = self.norm_l2(h2)

        # JK fusion (fuse h_af from layer 1 attention and h2 from layer 2)
        if self.use_jk:
            jk_out = self.jk([h_af, h2])
            out = self.jk_proj(jk_out)
            out = F.celu(out)
            out = self.norm_l3(out)
        else:
            out = h2
            out = F.celu(out)
            out = self.norm_l3(out)

        return out, attn_weights
