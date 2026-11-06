# -*- coding: utf-8 -*-
"""FBCNet: filter-bank convolutional network.

Adapted from https://github.com/ravikiran-mane/FBCNet (codes/centralRepo/networks.py;
MIT License, Copyright (c) 2020 ravikiran-mane; see third_party_licenses/FBCNet_LICENSE.txt).
Mane et al., "FBCNet: A multi-view convolutional neural network for brain-computer
interface", arXiv:2104.01233 (2021).

Changes with respect to the upstream code:
  1. The input is the already filtered 4-D tensor (batch, n_bands, channels, time) built by
     ``calibration_budget.filterbank.fbcnet_input``; the upstream 5-D permute/squeeze is
     not needed.
  2. The final LogSoftmax is removed; ``forward`` returns logits.
  3. Only the default temporal layer (LogVarLayer) is ported.
"""
import torch
import torch.nn as nn


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super().__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(x)


class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super().__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(x)


class swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class LogVarLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.log(torch.clamp(x.var(dim=self.dim, keepdim=True), 1e-6, 1e6))


class FBCNet(nn.Module):
    def __init__(self, n_classes, channels, samples,
                 nBands=9, m=32, strideFactor=4,
                 temporalLayer='LogVarLayer', doWeightNorm=True):
        super().__init__()
        assert temporalLayer == 'LogVarLayer', 'only the LogVarLayer temporal layer is ported'
        self.nBands = nBands
        self.m = m
        self.strideFactor = strideFactor

        self.scb = nn.Sequential(
            Conv2dWithConstraint(nBands, m * nBands, (channels, 1), groups=nBands,
                                 max_norm=2, doWeightNorm=doWeightNorm, padding=0),
            nn.BatchNorm2d(m * nBands),
            swish(),
        )
        self.temporalLayer = LogVarLayer(dim=3)
        self.lastLayer = LinearWithConstraint(
            self.m * self.nBands * self.strideFactor, n_classes,
            max_norm=0.5, doWeightNorm=doWeightNorm)

    def forward(self, x):
        # x: (batch, nBands, channels, time); time must be a multiple of strideFactor.
        assert x.shape[3] % self.strideFactor == 0, (
            f'time length {x.shape[3]} is not a multiple of strideFactor={self.strideFactor}')
        x = self.scb(x)                                        # (batch, m*nBands, 1, T)
        x = x.reshape([*x.shape[0:2], self.strideFactor, int(x.shape[3] / self.strideFactor)])
        x = self.temporalLayer(x)                              # (batch, m*nBands, strideFactor, 1)
        x = torch.flatten(x, start_dim=1)
        x = self.lastLayer(x)
        return x
