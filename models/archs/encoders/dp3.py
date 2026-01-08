#!/usr/bin/env python3

import torch
import torch.nn as nn

class DP3Encoder(nn.Module):
    """
        DP3 Encoder as defined in the paper 3D Diffusion Policy: Generalizable Visuomotor Policy Learning via Simple 3D Representations
        https://arxiv.org/abs/2403.03954

        Taken from Appendix of paper, section A: Implementation Details
    """
    def __init__(self, channels=3):
        # We only use xyz (channels=3) in this work
        # while our encoder also works for xyzrgb (channels=6) in our experiments
        self.mlp = nn.Sequential(
        nn.Linear(channels, 64), nn.LayerNorm(64), nn.ReLU(),
        nn.Linear(64, 128), nn.LayerNorm(128), nn.ReLU(),
        nn.Linear(128, 256), nn.LayerNorm(256), nn.ReLU())
        self.projection = nn.Sequential(nn.Linear(256, 64), nn.LayerNorm(64))

    def forward(self, x):
        # x: B, N, 3
        x = self.mlp(x) # B, N, 256
        x = torch.max(x, 1)[0] # B, 256
        x = self.projection(x) # B, 64
        return x
