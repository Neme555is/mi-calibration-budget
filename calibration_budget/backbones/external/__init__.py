# -*- coding: utf-8 -*-
"""Backbones whose source code is not redistributed with this repository.

EEG Conformer (GPL-3.0) and LMDA-Net (no license file upstream) must be downloaded from their
authors' repositories and placed in this folder; see ``README.md`` here for the exact steps.
"""
import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_README = os.path.join(HERE, 'README.md')


def _import(module):
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as exc:
        if exc.name == module:
            raise ImportError(f'{module}.py is missing from {HERE}. Follow {_README} to '
                              f'download it from the upstream repository.') from None
        if exc.name == 'torchsummary':
            raise ImportError('lmda_model.py imports torchsummary: either `pip install '
                              'torchsummary` or delete that first line of lmda_model.py.') from None
        raise


def build_conformer(n_classes, channels, samples):
    """EEG Conformer adapted as described in README.md (returns logits)."""
    return _import('conformer').Conformer(n_classes=n_classes, channels=channels,
                                          samples=samples)


def build_lmdanet(n_classes, channels, samples):
    """Upstream LMDA-Net, used unchanged with its default hyperparameters."""
    return _import('lmda_model').LMDA(chans=channels, samples=samples, num_classes=n_classes)
