# -*- coding: utf-8 -*-
"""EEGNet (Lawhern et al., J. Neural Eng. 2018), PyTorch implementation of Table 2 of the
paper. The μ-band CNN of this work is this network trained on 8–13 Hz input."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2dWithConstraint(nn.Conv2d):
    """Conv2d with a max-norm constraint on each filter (depthwise conv: max_norm = 1)."""

    def __init__(self, *args, max_norm=1.0, **kwargs):
        self.max_norm = max_norm
        super().__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(x)


class LinearWithConstraint(nn.Linear):
    """Linear layer with a max-norm constraint (classifier: max_norm = 0.25)."""

    def __init__(self, *args, max_norm=0.25, **kwargs):
        self.max_norm = max_norm
        super().__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(x)


class EEGNet(nn.Module):
    """
    Parameters
    ----------
    n_classes : number of classes
    channels : number of EEG channels
    samples : number of time samples
    F1 : number of temporal filters
    D : spatial filters per temporal filter
    F2 : pointwise filters (default D * F1)
    kernel_len : temporal kernel length (32 at 128 Hz with a 4 Hz high-pass)
    dropout : dropout rate (0.25 for cross-subject training)
    """

    def __init__(self, n_classes=4, channels=22, samples=448,
                 F1=8, D=2, F2=None, kernel_len=64, dropout=0.5):
        super().__init__()
        F2 = F2 or F1 * D

        # Block 1: temporal convolution, then depthwise spatial convolution.
        self.conv_temporal = nn.Conv2d(1, F1, kernel_size=(1, kernel_len),
                                       padding=(0, kernel_len // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.conv_spatial = Conv2dWithConstraint(F1, F1 * D, kernel_size=(channels, 1),
                                                 groups=F1, bias=False, max_norm=1.0)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)

        # Block 2: separable convolution (depthwise (1, 16) + pointwise (1, 1)).
        self.conv_sep_depth = nn.Conv2d(F1 * D, F1 * D, kernel_size=(1, 16),
                                        padding=(0, 8), groups=F1 * D, bias=False)
        self.conv_sep_point = nn.Conv2d(F1 * D, F2, kernel_size=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout)

        # Classifier: two poolings (4 and 8) leave samples // 32 time steps.
        flatten_size = F2 * (samples // 32)
        self.classifier = LinearWithConstraint(flatten_size, n_classes, max_norm=0.25)

    def forward(self, x, return_features=False):
        # x: (B, 1, C, T)
        x = self.bn1(self.conv_temporal(x))
        x = self.bn2(self.conv_spatial(x))
        x = F.elu(x)
        x = self.drop1(self.pool1(x))

        x = self.conv_sep_depth(x)
        x = self.bn3(self.conv_sep_point(x))
        x = F.elu(x)
        x = self.drop2(self.pool2(x))

        features = x.flatten(start_dim=1)
        logits = self.classifier(features)
        if return_features:
            return logits, features
        return logits
