# -*- coding: utf-8 -*-
"""Leave-one-subject-out data of the four motor-imagery datasets.

Every loader returns {subject: {X_train, y_train, X_test, y_test}} with X of shape
(n_trials, 1, n_channels, n_samples). For a target subject, X_train/y_train is its
calibration block and X_test/y_test its evaluation block; source subjects contribute both
blocks to the training pool.
"""
import os

from .. import config
from . import bci_iv_2a, cho2017, openbmi, physionet_mi

LOADERS = {'2a': bci_iv_2a, 'cho2017': cho2017, 'physionet': physionet_mi, 'openbmi': openbmi}
OPENBMI_CACHE = os.environ.get('OPENBMI_CACHE', config.results_path('cache', 'openbmi'))


def load_subjects(dataset, fmin=4, fmax=40):
    """All subjects of ``dataset`` band-passed to [fmin, fmax] Hz (not yet aligned)."""
    mod = LOADERS[dataset]
    if dataset == 'openbmi':
        return mod.load_all_subjects(subjects=mod.ALL_SUBJECTS, fmin=fmin, fmax=fmax,
                                     cache_dir=OPENBMI_CACHE)
    return mod.load_all_subjects(subjects=mod.ALL_SUBJECTS, fmin=fmin, fmax=fmax)
