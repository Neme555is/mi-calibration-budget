# -*- coding: utf-8 -*-
"""DeepConvNet (Deep4Net; Schirrmeister et al., Hum. Brain Mapp. 2017, Fig. 1).

Four convolution–max-pooling blocks; the first block splits the convolution into a
temporal and a spatial part without a nonlinearity in between. ELU, batch norm and dropout
follow each block.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepConvNet(nn.Module):
    """
    Parameters
    ----------
    n_classes, channels, samples : output classes, EEG channels, time samples
    F1, F2, F3, F4 : filters of blocks 1–4 (25, 50, 100, 200)
    kernel_len : temporal kernel length (10)
    pool_len, pool_stride : max-pooling window and stride (3, 3)
    dropout : dropout rate
    bn_momentum : batch-norm momentum (0.1)
    """

    def __init__(self, n_classes=4, channels=22, samples=448, F1=25, F2=50, F3=100, F4=200,
                 kernel_len=10, pool_len=3, pool_stride=3, dropout=0.5, bn_momentum=0.1):
        super().__init__()
        self.channels = channels
        self.samples = samples
        self.pool_stride = pool_stride
        self.kernel_len = kernel_len

        # Block 1: temporal convolution, then a full (non-depthwise) spatial convolution.
        self.conv_time = nn.Conv2d(1, F1, kernel_size=(1, kernel_len),
                                   padding=(0, kernel_len // 2), bias=False)
        self.conv_spat = nn.Conv2d(F1, F1, kernel_size=(channels, 1), padding=(0, 0),
                                   bias=False)
        self.bn1 = nn.BatchNorm2d(F1, momentum=bn_momentum)
        self.pool1 = nn.MaxPool2d((1, pool_len), stride=(1, pool_stride))
        self.drop1 = nn.Dropout2d(dropout)

        # Blocks 2–4.
        self.conv2 = nn.Conv2d(F1, F2, kernel_size=(1, kernel_len),
                               padding=(0, kernel_len // 2), bias=False)
        self.bn2 = nn.BatchNorm2d(F2, momentum=bn_momentum)
        self.pool2 = nn.MaxPool2d((1, pool_len), stride=(1, pool_stride))
        self.drop2 = nn.Dropout2d(dropout)

        self.conv3 = nn.Conv2d(F2, F3, kernel_size=(1, kernel_len),
                               padding=(0, kernel_len // 2), bias=False)
        self.bn3 = nn.BatchNorm2d(F3, momentum=bn_momentum)
        self.pool3 = nn.MaxPool2d((1, pool_len), stride=(1, pool_stride))
        self.drop3 = nn.Dropout2d(dropout)

        self.conv4 = nn.Conv2d(F3, F4, kernel_size=(1, kernel_len),
                               padding=(0, kernel_len // 2), bias=False)
        self.bn4 = nn.BatchNorm2d(F4, momentum=bn_momentum)
        self.pool4 = nn.MaxPool2d((1, pool_len), stride=(1, pool_stride))
        self.drop4 = nn.Dropout2d(dropout)

        self.flatten_size = self._compute_flatten_size()
        self.classifier = nn.Linear(self.flatten_size, n_classes)
        self._reset_parameters()

    def _compute_flatten_size(self):
        dummy = torch.zeros(1, 1, self.channels, self.samples)
        with torch.no_grad():
            x = self.conv_time(dummy)
            x = self.conv_spat(x)
            x = self.pool1(x)
            x = self.conv2(x)
            x = self.pool2(x)
            x = self.conv3(x)
            x = self.pool3(x)
            x = self.conv4(x)
            x = self.pool4(x)
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
        x = self.bn1(x)
        x = F.elu(x)
        x = self.drop1(self.pool1(x))

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.drop2(self.pool2(x))

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.drop3(self.pool3(x))

        x = self.conv4(x)
        x = self.bn4(x)
        x = F.elu(x)
        x = self.drop4(self.pool4(x))

        x = x.flatten(start_dim=1)
        return self.classifier(x)
