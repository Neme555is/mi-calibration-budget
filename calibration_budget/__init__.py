# -*- coding: utf-8 -*-
"""Calibration-budget analysis of cross-subject motor-imagery decoding.

Modules
-------
config        datasets, backbones, budget grids, result paths
data          loaders of BCI IV-2a, Cho2017, PhysioNet MI and OpenBMI MI
alignment     Euclidean alignment (EA)
filterbank    filter banks and per-channel normalization
backbones     the twelve backbones
training      leave-one-subject-out folds, source training, feature extraction
head          classifier-head fine-tuning on frozen features
curves        shared-exponent power law, crossover location
forms         alternative saturating forms and headroom
"""
