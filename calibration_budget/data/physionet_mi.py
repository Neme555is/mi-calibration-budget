# -*- coding: utf-8 -*-
"""PhysioNet EEG Motor Movement/Imagery (EEGMMIDB): left/right-hand imagery, 64 channels.

Imagery runs R4, R8 and R12 are used. Calibration block = R4 + R8, evaluation block = R12.
Subject 88 (recorded at 128 Hz instead of 160 Hz) is excluded, leaving 108 subjects.

Preprocessing (MOABB MotorImagery paradigm): band-pass [fmin, fmax] Hz, resampling to
128 Hz, window 0.5–4.0 s after the cue (449 samples).

If ``PHYSIONET_ROOT`` points to the EEGMMIDB folder (S001/S001R04.edf, …), those files are
used; otherwise MOABB downloads the dataset.
"""
import os
import types

import numpy as np
from moabb.datasets import PhysionetMI
from moabb.paradigms import MotorImagery

N_CLASSES = 2
NUM_CHANNELS = 64
LABEL_MAP = {'left_hand': 0, 'right_hand': 1}
CALIB_RUN_KEYS = {'0', '1'}      # R4 + R8 (MOABB run keys of the imagined-hand runs)
EVAL_RUN_KEYS = {'2'}            # R12
EXCLUDE_SUBJECTS = {88}
ALL_SUBJECTS = [s for s in range(1, 110) if s not in EXCLUDE_SUBJECTS]
LOCAL_ROOT = os.environ.get('PHYSIONET_ROOT')


def _dataset(subjects):
    ds = PhysionetMI(imagined=True, executed=False, subjects=list(subjects))
    if not LOCAL_ROOT:
        return ds

    def _load_data_local(self, subject, runs, path=None, force_update=False, verbose=None):
        runs = [runs] if not hasattr(runs, '__iter__') else list(runs)
        out = []
        for r in runs:
            f = os.path.join(LOCAL_ROOT, f'S{subject:03d}', f'S{subject:03d}R{r:02d}.edf')
            if not os.path.exists(f):
                raise FileNotFoundError(f'missing PhysioNet file: {f} (set PHYSIONET_ROOT)')
            out.append(f)
        return out

    ds._load_data = types.MethodType(_load_data_local, ds)
    cache = os.path.join(LOCAL_ROOT, 'mne_cache')        # MOABB checks that these paths exist
    os.makedirs(cache, exist_ok=True)
    os.environ['MNE_DATA'] = cache
    os.environ['MNE_DATASETS_EEGBCI_PATH'] = cache
    return ds


def load_all_subjects(subjects=None, fs_target=128, tmin=0.5, tmax=4.0, fmin=4, fmax=40):
    """Return {subject: {X_train, y_train, X_test, y_test}} with X of shape (n, 1, 64, 449)
    (float32); X_train/y_train is the calibration block (R4 + R8), X_test/y_test the
    evaluation block (R12)."""
    subjects = [s for s in (subjects or ALL_SUBJECTS) if s not in EXCLUDE_SUBJECTS]
    dataset = _dataset(subjects)
    paradigm = MotorImagery(n_classes=N_CLASSES, events=['left_hand', 'right_hand'],
                            fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax, resample=fs_target)
    out = {}
    for sid in subjects:
        X, y, meta = paradigm.get_data(dataset=dataset, subjects=[sid])
        calib_mask = meta['run'].isin(CALIB_RUN_KEYS).values
        eval_mask = meta['run'].isin(EVAL_RUN_KEYS).values
        if calib_mask.sum() == 0 or eval_mask.sum() == 0:
            raise RuntimeError(f'subject {sid}: unexpected runs {meta["run"].unique().tolist()}')
        y_idx = np.array([LABEL_MAP[v] for v in y])
        out[sid] = {
            'X_train': X[calib_mask][:, None, :, :].astype(np.float32),
            'y_train': y_idx[calib_mask].astype(np.int64),
            'X_test': X[eval_mask][:, None, :, :].astype(np.float32),
            'y_test': y_idx[eval_mask].astype(np.int64),
        }
    return out
