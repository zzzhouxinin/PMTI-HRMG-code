#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn

class DAE(nn.Module):
    def __init__(self, col, D, noise_std = 0.1):
        super().__init__()
        self.noise_std = noise_std
        self.encoder = nn.Sequential(
            nn.Linear(col, 2048),
            nn.Tanh(),
            nn.Linear(2048, 1536),
            nn.Tanh(),
            nn.Linear(1536, 1024),
            nn.Tanh(),
            nn.Linear(1024, D),
            nn.Tanh()
            )
        
        self.decoder = nn.Sequential(
            nn.Linear(D, 1024),
            nn.Tanh(),
            nn.Linear(1024, 1536),
            nn.Tanh(),
            nn.Linear(1536, 2048),
            nn.Tanh(),
            nn.Linear(2048, col),
            nn.Tanh(),
            )
        
    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.noise_std
            x_noisy = x + noise
        else:
            x_noisy = x
        encoded = self.encoder(x_noisy)
        decoded = self.decoder(encoded)
        return encoded, decoded
