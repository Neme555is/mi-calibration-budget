# -*- coding: utf-8 -*-
"""The twelve backbones and their fixed hyperparameters.

Every backbone is built from the shapes of the training array, so the same definition
serves 22-channel (BCI IV-2a), 62-channel (OpenBMI) and 64-channel (Cho2017, PhysioNet)
inputs. ``KIND`` selects the training routine in ``calibration_budget.training``:

  cnn     gradient training with Adam (``train_network``)
  fbcnet  as ``cnn``, on the nine-band input of ``filterbank.fbcnet_input``
  tsmnet  as ``cnn`` with Riemannian Adam, on the CPU, 3-D input
  scsp    closed-form spectral CSP pipeline (``spectral_csp.fit_fold``)
  fbcsp   closed-form FBCSP pipeline (``fbcsp.fit_fold``)

EEG Conformer and LMDA-Net are not redistributed; ``external/README.md`` explains how to
obtain them.
"""
from .ctnet import CTNet
from .deepconvnet import DeepConvNet
from .eegnet import EEGNet
from .eegtcnet import EEGTCNet
from .external import build_conformer, build_lmdanet
from .fbcnet import FBCNet
from .shallowconvnet import ShallowConvNet
from .tsmnet import build_tsmnet

KIND = {
    'mu_cnn': 'cnn', 'scsp': 'scsp', 'fbcsp': 'fbcsp', 'fbcnet': 'fbcnet',
    'shallowconvnet': 'cnn', 'deepconvnet': 'cnn', 'ctnet': 'cnn', 'conformer': 'cnn',
    'eegtcnet': 'cnn', 'lmdanet': 'cnn', 'tsmnet': 'tsmnet', 'eegnet': 'cnn',
}


def _eegnet(n_classes, X_tr):
    # EEGNet-8,2 (Lawhern et al., J. Neural Eng. 2018); the μ-band CNN is the same network
    # trained on 8–13 Hz input.
    return EEGNet(n_classes=n_classes, channels=X_tr.shape[2], samples=X_tr.shape[3],
                  F1=8, D=2, kernel_len=32, dropout=0.25)


def _shallowconvnet(n_classes, X_tr):
    return ShallowConvNet(n_classes=n_classes, channels=X_tr.shape[2], samples=X_tr.shape[3],
                          F1=40, F2=40, kernel_len=25, pool_len=75, pool_stride=15,
                          dropout=0.25, bn_momentum=0.1)


def _deepconvnet(n_classes, X_tr):
    return DeepConvNet(n_classes=n_classes, channels=X_tr.shape[2], samples=X_tr.shape[3],
                       F1=25, F2=50, F3=100, F4=200, kernel_len=10,
                       pool_len=3, pool_stride=3, dropout=0.25, bn_momentum=0.1)


def _ctnet(n_classes, X_tr):
    return CTNet(n_classes=n_classes, channels=X_tr.shape[2],
                 heads=2, emb_size=16, depth=6, f1=8, kernel_size=64, D=2,
                 pooling_size1=8, pooling_size2=8, dropout_rate=0.3)


def _conformer(n_classes, X_tr):
    return build_conformer(n_classes, channels=X_tr.shape[2], samples=X_tr.shape[3])


def _eegtcnet(n_classes, X_tr):
    return EEGTCNet(n_classes=n_classes, channels=X_tr.shape[2], samples=X_tr.shape[3])


def _lmdanet(n_classes, X_tr):
    return build_lmdanet(n_classes, channels=X_tr.shape[2], samples=X_tr.shape[3])


def _fbcnet(n_classes, X_tr):
    # X_tr is the nine-band input (n, 9, ch, t).
    return FBCNet(n_classes=n_classes, channels=X_tr.shape[2], samples=X_tr.shape[3])


def _tsmnet(n_classes, X_tr):
    # X_tr is 3-D (n, ch, t).
    return build_tsmnet(n_classes, channels=X_tr.shape[1], samples=X_tr.shape[2])


BUILDERS = {
    'mu_cnn': _eegnet, 'eegnet': _eegnet, 'shallowconvnet': _shallowconvnet,
    'deepconvnet': _deepconvnet, 'ctnet': _ctnet, 'conformer': _conformer,
    'eegtcnet': _eegtcnet, 'lmdanet': _lmdanet, 'fbcnet': _fbcnet, 'tsmnet': _tsmnet,
}


def build(backbone, n_classes, X_tr):
    """Instantiate a gradient-trained backbone for training array ``X_tr``."""
    return BUILDERS[backbone](n_classes, X_tr)
