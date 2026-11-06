# -*- coding: utf-8 -*-
"""TSMNet: tangent-space mapping network with SPD batch normalization.

The implementation in ``tsmnet_lib/`` is vendored from https://github.com/rkobler/TSMNet
(spdnets/functionals.py, manifolds.py, modules.py, batchnorm.py, models/base.py and
models/tsmnet.py; BSD 3-Clause License, Copyright (c) 2022, rkobler; see
``tsmnet_lib/LICENSE``). Only the import statements were changed to fit a flat package
layout; the numerical code is unchanged. Kobler et al., "SPD domain-specific batch
normalization to crack interpretable unsupervised domain adaptation in EEG", NeurIPS 2022.

The SPD part of the network runs on the CPU in double precision (hard-coded upstream).
Inputs are 3-D, (batch, channels, time); ``forward(x, d=None, return_latent=False)``
returns logits. No domain-specific batch normalization is used (``domains=[]``).
"""
import torch

from .tsmnet_lib.tsmnet import TSMNet

__all__ = ['TSMNet', 'build_tsmnet']


def build_tsmnet(n_classes, channels, samples):
    return TSMNet(nclasses=n_classes, nchannels=channels, nsamples=samples,
                  temporal_filters=4, spatial_filters=40, subspacedims=20,
                  temp_cnn_kernel=25, bnorm='spdbn',
                  device=torch.device('cpu'), domains=[])
