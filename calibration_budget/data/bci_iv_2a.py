# -*- coding: utf-8 -*-
"""BCI Competition IV dataset 2a (BNCI 2014-001): 9 subjects, 4 classes, 22 channels.

Calibration block = session 1 ("0train"); evaluation block = session 2 ("1test").
Preprocessing (MOABB MotorImagery paradigm): band-pass [fmin, fmax] Hz, resampling to
128 Hz, window 0.5–4.0 s after the cue.

Data are read through MOABB. If the environment variable ``BCI2A_ROOT`` points to a folder
holding A01T.mat … A09E.mat, those files are used and nothing is downloaded; otherwise
MOABB downloads the dataset into its default location.
"""
import os
import re

import numpy as np
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery

N_CLASSES = 4
ALL_SUBJECTS = list(range(1, 10))
LABEL_MAP = {'left_hand': 0, 'right_hand': 1, 'feet': 2, 'tongue': 3}
LOCAL_ROOT = os.environ.get('BCI2A_ROOT')


def _use_local_files():
    """Redirect MOABB's BNCI file lookup to the local .mat files (idempotent)."""
    from moabb.datasets.bnci import bnci_2014

    if getattr(bnci_2014, '_local_patched', False):
        return

    def _data_path_local(url, path=None, force_update=False, update_path=None, verbose=None):
        m = re.search(r'A(\d{2})([TE])\.mat', str(url))
        if m is None:
            raise ValueError(f'cannot derive a local file name from URL: {url}')
        f = os.path.join(LOCAL_ROOT, f'A{m.group(1)}{m.group(2)}.mat')
        if not os.path.exists(f):
            raise FileNotFoundError(f'missing BCI IV-2a file: {f} (set BCI2A_ROOT)')
        return [f]

    bnci_2014.data_path = _data_path_local
    bnci_2014._local_patched = True


def load_all_subjects(subjects=None, fs_target=128, tmin=0.5, tmax=4.0, fmin=4, fmax=40):
    """Return {subject: {X_train, y_train, X_test, y_test}} with X of shape (n, 1, 22, T)
    (float32); X_train/y_train is the calibration block, X_test/y_test the evaluation block."""
    if LOCAL_ROOT:
        _use_local_files()
    paradigm = MotorImagery(n_classes=N_CLASSES, fmin=fmin, fmax=fmax,
                            tmin=tmin, tmax=tmax, resample=fs_target)
    dataset = BNCI2014_001()
    out = {}
    for sid in (subjects or ALL_SUBJECTS):
        X, y, meta = paradigm.get_data(dataset=dataset, subjects=[sid])
        train_mask = meta['session'] == '0train'
        test_mask = meta['session'] == '1test'
        y_idx = np.array([LABEL_MAP[label] for label in y])
        out[sid] = {
            'X_train': X[train_mask][:, None, :, :].astype(np.float32),
            'y_train': y_idx[train_mask],
            'X_test': X[test_mask][:, None, :, :].astype(np.float32),
            'y_test': y_idx[test_mask],
        }
    return out
