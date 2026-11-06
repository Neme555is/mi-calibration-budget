# -*- coding: utf-8 -*-
"""OpenBMI motor imagery (Lee et al., GigaScience 2019): 54 subjects × 2 days, 62 channels.

Each day has an offline block (EEG_MI_train, 100 trials) and an online-feedback block
(EEG_MI_test). Only the offline blocks are used: during the online block the fixation cross
moved with the output of a real-time classifier, so the displayed feedback is correlated
with the label. Calibration block = day-1 offline block; evaluation block = day-2 offline
block (50 trials per class each). The native 62 channels are kept.

Preprocessing (MOABB MotorImagery paradigm): band-pass [fmin, fmax] Hz, resampling to
128 Hz, window 0.5–4.0 s after the cue (448 samples).

MOABB 1.5 filters the sessions returned by ``Lee2019_MI`` with its own session names
("1", "2") while the dataset labels them "0", "1", which silently drops day 1. The loader
disables that filter so both days are returned.

If ``OPENBMI_ROOT`` points to the folder with sess01_subj01_EEG_MI.mat, …, those files are
used; otherwise MOABB downloads the dataset. Loading 54 subjects at 1000 Hz is slow, so an
optional on-disk cache (``cache_dir``) stores the preprocessed arrays.
"""
import os
import types

import numpy as np
from moabb.datasets import Lee2019_MI
from moabb.paradigms import MotorImagery

N_CLASSES = 2
NUM_CHANNELS = 62
LABEL_MAP = {'left_hand': 0, 'right_hand': 1}
ALL_SUBJECTS = list(range(1, 55))
FILE_FMT = 'sess{session:02d}_subj{subject:02d}_EEG_{suffix}.mat'
LOCAL_ROOT = os.environ.get('OPENBMI_ROOT')


def _dataset(subjects):
    ds = Lee2019_MI(subjects=list(subjects), sessions=(1, 2))
    ds._selected_sessions = None             # keep both days (see module docstring)
    if not LOCAL_ROOT:
        return ds

    def _data_path_local(self, subject, path=None, force_update=False, update_path=None,
                         verbose=None):
        if subject not in self.subject_list:
            raise ValueError(f'invalid subject {subject}')
        out = []
        for session in self.sessions:
            f = os.path.join(LOCAL_ROOT, FILE_FMT.format(session=session, subject=subject,
                                                         suffix=self.code_suffix))
            if not os.path.exists(f):
                raise FileNotFoundError(f'missing OpenBMI file: {f} (set OPENBMI_ROOT)')
            out.append(f)
        return out

    ds.data_path = types.MethodType(_data_path_local, ds)
    cache = os.path.join(LOCAL_ROOT, 'mne_cache')        # MOABB checks that MNE_DATA exists
    os.makedirs(cache, exist_ok=True)
    os.environ['MNE_DATA'] = cache
    return ds


def _split_sessions(X, y, meta, sid):
    """Day 1 → calibration block, day 2 → evaluation block (by MOABB's session label)."""
    if 'session' not in meta.columns:
        raise RuntimeError(f'subject {sid}: MOABB metadata has no session column')
    sess = np.asarray(meta['session'].values)
    uniq = sorted(set(sess.tolist()))
    if len(uniq) != 2:
        raise RuntimeError(f'subject {sid}: expected 2 sessions, got {uniq}')
    m_calib, m_eval = sess == uniq[0], sess == uniq[1]
    return (X[m_calib], y[m_calib]), (X[m_eval], y[m_eval])


def _load_uncached(subjects, fs_target, tmin, tmax, fmin, fmax):
    dataset = _dataset(subjects)
    paradigm = MotorImagery(n_classes=N_CLASSES, events=['left_hand', 'right_hand'],
                            fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax, resample=fs_target)
    out = {}
    for sid in subjects:
        X, y, meta = paradigm.get_data(dataset=dataset, subjects=[sid])
        X = X[:, None, :, :].astype(np.float32)
        y_idx = np.array([LABEL_MAP[v] for v in y]).astype(np.int64)
        (X_calib, y_calib), (X_eval, y_eval) = _split_sessions(X, y_idx, meta, sid)
        out[sid] = {'X_train': X_calib, 'y_train': y_calib, 'X_test': X_eval, 'y_test': y_eval}
    return out


def load_all_subjects(subjects=None, fs_target=128, tmin=0.5, tmax=4.0, fmin=4, fmax=40,
                      cache_dir=None):
    """Return {subject: {X_train, y_train, X_test, y_test}} with X of shape (100, 1, 62, 448)
    (float32); X_train/y_train is the day-1 block, X_test/y_test the day-2 block.

    With ``cache_dir``, the arrays of each (band, subject set) are stored in one .npz file
    and reused on later calls.
    """
    subjects = list(subjects or ALL_SUBJECTS)
    if cache_dir is None:
        return _load_uncached(subjects, fs_target, tmin, tmax, fmin, fmax)
    os.makedirs(cache_dir, exist_ok=True)
    tag = f'ob_{fmin:g}_{fmax:g}_{fs_target}_{len(subjects)}_{min(subjects)}_{max(subjects)}'
    cpath = os.path.join(cache_dir, tag + '.npz')
    if os.path.exists(cpath):
        z = np.load(cpath)
        return {sid: dict(X_train=z[f's{sid}|Xtr'], y_train=z[f's{sid}|ytr'],
                          X_test=z[f's{sid}|Xte'], y_test=z[f's{sid}|yte'])
                for sid in (int(x) for x in z['__sids__'])}
    out = _load_uncached(subjects, fs_target, tmin, tmax, fmin, fmax)
    np.savez(cpath, __sids__=np.array(sorted(out), int), **{
        f's{sid}|{p}': out[sid][k] for sid in out
        for p, k in (('Xtr', 'X_train'), ('ytr', 'y_train'), ('Xte', 'X_test'), ('yte', 'y_test'))})
    return out
