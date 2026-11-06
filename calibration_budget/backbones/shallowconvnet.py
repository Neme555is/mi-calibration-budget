# -*- coding: utf-8 -*-
"""ShallowConvNet (Schirrmeister et al., Hum. Brain Mapp. 2017, Fig. 2).

Temporal convolution → spatial convolution → batch norm → squaring → mean pooling → log,
i.e. a learned analogue of filter-bank log-variance features.
"""
import torch
import torch.nn as nn


class ShallowConvNet(nn.Module):
    """
    Parameters
    ----------
    n_classes, channels, samples : output classes, EEG channels, time samples
    F1 : temporal filters (40)
    F2 : spatial filters (40)
    kernel_len : temporal kernel length (25)
    pool_len, pool_stride : mean-pooling window and stride (75, 15)
    dropout : dropout rate
    bn_momentum : batch-norm momentum
    """

    def __init__(self, n_classes=4, channels=22, samples=448, F1=40, F2=40,
                 kernel_len=25, pool_len=75, pool_stride=15, dropout=0.5, bn_momentum=0.1):
        super().__init__()
        self.channels = channels
        self.samples = samples

        self.conv_time = nn.Conv2d(1, F1, kernel_size=(1, kernel_len),
                                   padding=(0, kernel_len // 2), bias=False)
        self.conv_spat = nn.Conv2d(F1, F2, kernel_size=(channels, 1), padding=(0, 0),
                                   bias=False)
        self.bn = nn.BatchNorm2d(F2, momentum=bn_momentum)
        self.pool = nn.AvgPool2d((1, pool_len), stride=(1, pool_stride))
        self.drop = nn.Dropout(dropout)

        self.flatten_size = self._compute_flatten_size()
        self.classifier = nn.Linear(self.flatten_size, n_classes)
        self._reset_parameters()

    def _compute_flatten_size(self):
        dummy = torch.zeros(1, 1, self.channels, self.samples)
        with torch.no_grad():
            x = self.conv_time(dummy)
            x = self.conv_spat(x)
            x = x.pow(2)
            x = self.pool(x)
            x = torch.log(x + 1e-8)
            return x.numel()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B, 1, C, T)
        x = self.conv_time(x)
        x = self.conv_spat(x)
        x = self.bn(x)
        x = x.pow(2)
        x = self.pool(x)
        x = torch.log(x + 1e-8)
        x = self.drop(x)
        x = x.flatten(start_dim=1)
        return self.classifier(x)
