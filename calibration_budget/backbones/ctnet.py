# -*- coding: utf-8 -*-
"""CTNet: EEGNet-style convolutional patch embedding + Transformer encoder.

Adapted from https://github.com/snailpt/CTNet (MIT License, Copyright (c) 2024 snailpt;
see third_party_licenses/CTNet_LICENSE.txt). Zhao, Jiang, Zhang, Xiao and Weng,
"CTNet: a convolutional transformer network for EEG-based motor imagery classification",
Sci. Rep. 14, 20237 (2024).

Changes with respect to the upstream code:
  1. The positional encoding uses ``x.device`` instead of a hard-coded ``.cuda()`` call.
  2. The classification head uses ``nn.LazyLinear`` so that any input length works
     (upstream hard-codes the flattened size of 1000-sample, 250 Hz inputs).
  3. ``forward`` returns logits only.
Architecture hyperparameters are the upstream defaults (heads 2, depth 6, embedding 16,
F1 8, D 2, pooling 8/8, dropout 0.3, kernel 64). At the 128 Hz sampling rate used here the
64-sample temporal kernel spans 0.5 s (0.256 s at the upstream 250 Hz).
"""
import math

import torch
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import Tensor, nn


class PatchEmbeddingCNN(nn.Module):
    def __init__(self, f1=8, kernel_size=64, D=2, pooling_size1=8, pooling_size2=8,
                 dropout_rate=0.3, number_channel=22, emb_size=16):
        super().__init__()
        f2 = D * f1
        self.cnn_module = nn.Sequential(
            nn.Conv2d(1, f1, (1, kernel_size), (1, 1), padding='same', bias=False),
            nn.BatchNorm2d(f1),
            nn.Conv2d(f1, f2, (number_channel, 1), (1, 1), groups=f1, padding='valid', bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, pooling_size1)),
            nn.Dropout(dropout_rate),
            nn.Conv2d(f2, f2, (1, 16), padding='same', bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, pooling_size2)),
            nn.Dropout(dropout_rate),
        )
        self.projection = nn.Sequential(Rearrange('b e (h) (w) -> b (h w) e'))

    def forward(self, x: Tensor) -> Tensor:
        x = self.cnn_module(x)
        x = self.projection(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        q = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', q, k)
        if mask is not None:
            energy = energy.masked_fill(~mask, torch.finfo(torch.float32).min)
        att = F.softmax(energy / (self.emb_size ** 0.5), dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav', att, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.projection(out)


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class ResidualAdd(nn.Module):
    def __init__(self, fn, emb_size, drop_p):
        super().__init__()
        self.fn = fn
        self.drop = nn.Dropout(drop_p)
        self.layernorm = nn.LayerNorm(emb_size)

    def forward(self, x, **kwargs):
        return self.layernorm(self.drop(self.fn(x, **kwargs)) + x)


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size, num_heads=4, drop_p=0.5, forward_expansion=4, forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(MultiHeadAttention(emb_size, num_heads, drop_p)),
                        emb_size, drop_p),
            ResidualAdd(nn.Sequential(
                FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p)),
                emb_size, drop_p),
        )


class TransformerEncoder(nn.Sequential):
    def __init__(self, heads, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size, heads) for _ in range(depth)])


class BranchEEGNetTransformer(nn.Sequential):
    def __init__(self, heads=2, depth=6, emb_size=16, number_channel=22, f1=8,
                 kernel_size=64, D=2, pooling_size1=8, pooling_size2=8,
                 dropout_rate=0.3, **kwargs):
        super().__init__(PatchEmbeddingCNN(
            f1=f1, kernel_size=kernel_size, D=D, pooling_size1=pooling_size1,
            pooling_size2=pooling_size2, dropout_rate=dropout_rate,
            number_channel=number_channel, emb_size=emb_size))


class PositioinalEncoding(nn.Module):
    """Learnable positional embedding (class name as in the upstream code)."""

    def __init__(self, embedding, length=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.encoding = nn.Parameter(torch.randn(1, length, embedding))

    def forward(self, x):
        x = x + self.encoding[:, :x.shape[1], :].to(x.device)
        return self.dropout(x)


class CTNet(nn.Module):
    """CNN patch embedding + Transformer encoder + linear head; returns logits."""

    def __init__(self, n_classes=4, channels=22, heads=2, emb_size=16, depth=6,
                 f1=8, kernel_size=64, D=2, pooling_size1=8, pooling_size2=8,
                 dropout_rate=0.3, **kwargs):
        super().__init__()
        self.emb_size = emb_size
        self.cnn = BranchEEGNetTransformer(
            heads, depth, emb_size, number_channel=channels, f1=f1,
            kernel_size=kernel_size, D=D, pooling_size1=pooling_size1,
            pooling_size2=pooling_size2, dropout_rate=dropout_rate)
        self.position = PositioinalEncoding(emb_size, dropout=0.1)
        self.trans = TransformerEncoder(heads, depth, emb_size)
        self.flatten = nn.Flatten()
        self.classification = nn.Sequential(nn.Dropout(0.5), nn.LazyLinear(n_classes))

    def forward(self, x):
        cnn = self.cnn(x)
        cnn = cnn * math.sqrt(self.emb_size)
        cnn = self.position(cnn)
        trans = self.trans(cnn)
        features = cnn + trans
        return self.classification(self.flatten(features))
