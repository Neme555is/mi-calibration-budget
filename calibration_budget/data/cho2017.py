# -*- coding: utf-8 -*-
"""Cho2017 (GigaScience): 52 subjects, left/right-hand imagery, 64 EEG channels, 512 Hz.

The dataset has a single session per subject. Trials are kept in acquisition order and
split per class: the first 67 trials of each class form the calibration block (134 trials)
and the remaining trials the evaluation block (66 trials).

Channels are reordered to the 64-channel layout ``CH_NAMES_64`` shared with PhysioNet MI.
The two montages differ in two channels (Cho2017 has P9/P10 but no T9/T10); T9/T10 are
zero-filled.

Preprocessing (MOABB MotorImagery paradigm): band-pass [fmin, fmax] Hz, resampling to
128 Hz, window 0.5–4.0 s after the cue (448 samples).

If ``CHO2017_ROOT`` points to a folder with s01.mat … s52.mat, those files are used;
otherwise MOABB downloads the dataset.
"""
import os
import types

import numpy as np
from moabb.datasets import Cho2017
from moabb.paradigms import MotorImagery

N_CLASSES = 2
NUM_CHANNELS = 64
LABEL_MAP = {'left_hand': 0, 'right_hand': 1}
ALL_SUBJECTS = list(range(1, 53))
N_CALIB_PER_CLASS = 67
LOCAL_ROOT = os.environ.get('CHO2017_ROOT')

# EEG channel order of the MOABB Cho2017 recordings (BioSemi order).
_CHO2017_EEG_NAMES = [
    'Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1',
    'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7',
    'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz', 'Fpz', 'Fp2',
    'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4',
    'FC2', 'FCz', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2',
    'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2',
]
# Target 64-channel layout (sensorimotor channels first, then the remaining scalp sites).
CH_NAMES_64 = [
    'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6',
    'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
    'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6',
    'Fp1', 'Fpz', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8',
    'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
    'FT7', 'FT8', 'T7', 'T8', 'T9', 'T10', 'TP7', 'TP8',
    'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
    'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2', 'Iz',
]
_name2src = {n: i for i, n in enumerate(_CHO2017_EEG_NAMES)}
_REORDER = [_name2src.get(n, -1) for n in CH_NAMES_64]      # -1 → zero-filled channel
assert _REORDER[42] == -1 and _REORDER[43] == -1            # T9 / T10
assert all(i >= 0 for i in _REORDER[:21])                    # sensorimotor channels present


def _reorder_to_ch64(X):
    """(n, 64, t) in Cho2017 order → (n, 64, t) in CH_NAMES_64 order, T9/T10 zero-filled."""
    out = np.zeros((X.shape[0], NUM_CHANNELS, X.shape[2]), dtype=X.dtype)
    for pos, src in enumerate(_REORDER):
        if src >= 0:
            out[:, pos, :] = X[:, src, :]
    return out


def _dataset(subjects):
    ds = Cho2017(subjects=list(subjects))
    if not LOCAL_ROOT:
        return ds

    def _data_path_local(self, subject, path=None, force_update=False, update_path=None,
                         verbose=None):
        if subject not in self.subject_list:
            raise ValueError(f'invalid subject {subject}')
        f = os.path.join(LOCAL_ROOT, f's{subject:02d}.mat')
        if not os.path.exists(f):
            raise FileNotFoundError(f'missing Cho2017 file: {f} (set CHO2017_ROOT)')
        return f

    ds.data_path = types.MethodType(_data_path_local, ds)
    cache = os.path.join(LOCAL_ROOT, 'mne_cache')        # MOABB checks that MNE_DATA exists
    os.makedirs(cache, exist_ok=True)
    os.environ['MNE_DATA'] = cache
    return ds


def _split_calib_eval(X, y, n_calib_per_class=N_CALIB_PER_CLASS):
    """First ``n_calib_per_class`` trials of each class → calibration, the rest → evaluation."""
    y = np.asarray(y)
    calib_idx, eval_idx = [], []
    for cls in (0, 1):
        idx = np.where(y == cls)[0]
        if len(idx) < 2:
            raise RuntimeError(f'too few trials in class {cls}: {len(idx)}')
        n_cb = min(n_calib_per_class, len(idx) - 1)
        calib_idx.extend(idx[:n_cb].tolist())
        eval_idx.extend(idx[n_cb:].tolist())
    calib_idx = np.array(sorted(calib_idx))
    eval_idx = np.array(sorted(eval_idx))
    return X[calib_idx], y[calib_idx], X[eval_idx], y[eval_idx]


def load_all_subjects(subjects=None, fs_target=128, tmin=0.5, tmax=4.0, fmin=4, fmax=40):
    """Return {subject: {X_train, y_train, X_test, y_test}} with X of shape (n, 1, 64, 448)
    (float32); X_train/y_train is the calibration block, X_test/y_test the evaluation block."""
    subjects = list(subjects or ALL_SUBJECTS)
    dataset = _dataset(subjects)
    paradigm = MotorImagery(n_classes=N_CLASSES, events=['left_hand', 'right_hand'],
                            fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax, resample=fs_target)
    out = {}
    for sid in subjects:
        X, y, _meta = paradigm.get_data(dataset=dataset, subjects=[sid])
        X = _reorder_to_ch64(X)[:, None, :, :].astype(np.float32)
        y_idx = np.array([LABEL_MAP[v] for v in y]).astype(np.int64)
        X_calib, y_calib, X_eval, y_eval = _split_calib_eval(X, y_idx)
        out[sid] = {'X_train': X_calib, 'y_train': y_calib,
                    'X_test': X_eval, 'y_test': y_eval}
    return out
