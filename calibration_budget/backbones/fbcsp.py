# -*- coding: utf-8 -*-
"""FBCSP: filter-bank common spatial patterns (Ang et al., Front. Neurosci. 2012).

Nine 4-Hz bands (4–40 Hz) → per-band MNE CSP (6 components, OAS-regularized covariance,
log average power) through MOABB's ``FilterBank`` → standardization → selection of the
k = min(24, 9 · 6 · 2 − 1) features with the largest mutual information (random_state = 42)
→ LDA. The penultimate representation used for head fine-tuning is the selection output.
"""
from functools import partial

import numpy as np
from mne.decoding import CSP
from moabb.pipelines import FilterBank
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..filterbank import NINE_BANDS, filter_bank_stack, normalize_channels, to_3d

CSP_COMPONENTS = 6
SELECT_K = 24
MI_SEED = 42


class CSPWithLogVar(BaseEstimator, TransformerMixin):
    """MNE CSP with log average-power features, usable inside MOABB's FilterBank."""

    def __init__(self, n_components=6, reg='oas', log=True, norm_trace=False):
        self.n_components = n_components
        self.reg = reg
        self.log = log
        self.norm_trace = norm_trace

    def fit(self, X, y):
        self.csp_ = CSP(n_components=self.n_components, reg=self.reg, log=self.log,
                        norm_trace=self.norm_trace, transform_into='average_power')
        self.csp_.fit(X, y)
        return self

    def transform(self, X):
        return self.csp_.transform(X)


def fit_fold(X_tr, y_tr, X_calib, y_calib, X_eval, y_eval):
    """Fit on the source trials of one fold; return features and probabilities.

    X_* are EA-aligned (n, 1, ch, t) arrays. The validation split of the fold is not used.
    """
    X_tr, X_calib, X_eval = normalize_channels(X_tr, X_calib, X_eval)
    X_tr, X_calib, X_eval = to_3d(X_tr), to_3d(X_calib), to_3d(X_eval)
    fb_tr = filter_bank_stack(X_tr, NINE_BANDS)
    fb_calib = filter_bank_stack(X_calib, NINE_BANDS)
    fb_eval = filter_bank_stack(X_eval, NINE_BANDS)

    k = min(SELECT_K, fb_tr.shape[-1] * CSP_COMPONENTS * 2 - 1)
    pipeline = Pipeline([
        ('fbcsp', FilterBank(estimator=CSPWithLogVar(n_components=CSP_COMPONENTS, reg='oas',
                                                     log=True, norm_trace=False),
                             flatten=True)),
        ('scaler', StandardScaler()),
        ('select', SelectKBest(score_func=partial(mutual_info_classif, random_state=MI_SEED),
                               k=k)),
        ('classifier', LDA()),
    ])
    pipeline.fit(fb_tr, y_tr)
    pre = Pipeline(pipeline.steps[:-1])
    return {
        'train_feats': pre.transform(fb_tr).astype(np.float32), 'train_y': y_tr.astype(np.int64),
        'calib_feats': pre.transform(fb_calib).astype(np.float32),
        'calib_y': y_calib.astype(np.int64),
        'eval_feats': pre.transform(fb_eval).astype(np.float32), 'eval_y': y_eval.astype(np.int64),
        'train_proba_rerun': pipeline.predict_proba(fb_tr).astype(np.float32),
        'calib_proba_rerun': pipeline.predict_proba(fb_calib).astype(np.float32),
        'eval_proba_rerun': pipeline.predict_proba(fb_eval).astype(np.float32),
        'pipeline': pipeline,
    }
