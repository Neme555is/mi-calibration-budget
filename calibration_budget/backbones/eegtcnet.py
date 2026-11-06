# -*- coding: utf-8 -*-
"""EEG-TCNet: EEGNet front end + temporal convolutional network (PyTorch re-implementation).

Re-implemented from the Keras model in https://github.com/Thoriri/eeg-tcnet-1
(utils/models.py; Apache License 2.0). Ingolfsson et al., "EEG-TCNet: An accurate temporal
convolutional network for embedded motor-imagery brain–machine interfaces", IEEE SMC 2020.
Hyperparameters follow the repository's training example (F1 = 8, KE = 32, KT = 4, L = 2,
FT = 12, pe = 0.2, pt = 0.3).
"""
import torch.nn as nn


class EEGNetFrontEnd(nn.Module):
    """EEGNet blocks 1 and 2 of the Keras model."""

    def __init__(self, F1, D, kernLength, channels, dropout):
        super().__init__()
        F2 = F1 * D
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, kernLength), padding=(0, kernLength // 2), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F2, kernel_size=(channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(F2, F2, kernel_size=(1, 16), groups=F2, padding=(0, 8), bias=False),
            nn.Conv2d(F2, F2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )
        self.F2 = F2

    def forward(self, x):
        x = self.block1(x)      # (batch, F2, 1, T')
        x = self.block2(x)      # (batch, F2, 1, T'')
        x = x.squeeze(2)        # (batch, F2, T'')
        return x


class CausalConv1d(nn.Module):
    """Causal convolution: left padding of (kernel_size − 1)·dilation, extra outputs trimmed."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, **kwargs):
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              dilation=dilation, padding=self.left_pad, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        if self.left_pad > 0:
            x = x[:, :, :-self.left_pad]
        return x


class TCNResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        self.conv1 = CausalConv1d(in_ch, out_ch, kernel_size, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = CausalConv1d(out_ch, out_ch, kernel_size, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(dropout)
        self.act = nn.ELU()
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        res = self.downsample(x)
        h = self.act(self.bn1(self.conv1(x)))
        h = self.drop(h)
        h = self.bn2(self.conv2(h))
        return self.drop(self.act(h + res))


class TCNStack(nn.Module):
    def __init__(self, input_dim, depth, kernel_size, filters, dropout):
        super().__init__()
        blocks = []
        for i in range(depth):
            dilation = 2 ** i
            blocks.append(TCNResBlock(input_dim if i == 0 else filters,
                                      filters, kernel_size, dilation, dropout))
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)


class EEGTCNet(nn.Module):
    def __init__(self, n_classes, channels, samples,
                 layers=2, kernel_s=4, filt=12,
                 F1=8, D=2, kernLength=32, dropout=0.3, dropout_eeg=0.2):
        super().__init__()
        self.eegnet = EEGNetFrontEnd(F1=F1, D=D, kernLength=kernLength,
                                     channels=channels, dropout=dropout_eeg)
        self.tcn = TCNStack(input_dim=F1 * D, depth=layers, kernel_size=kernel_s,
                            filters=filt, dropout=dropout)
        self.classifier = nn.Linear(filt, n_classes)

    def forward(self, x):
        x = self.eegnet(x)       # (batch, F1*D, T'')
        x = self.tcn(x)          # (batch, filt, T'')
        x = x[:, :, -1]          # last time step
        return self.classifier(x)
