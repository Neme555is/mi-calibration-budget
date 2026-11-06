# -*- coding: utf-8 -*-
"""Spectral CSP: one-vs-rest CSP in four sensorimotor bands + mutual-information selection + LDA.

For every band (μ 8–13, low β 13–20, high β 20–30, μ+β 8–30 Hz) and every class c, the
spatial filters solve the generalized eigenproblem

    Σ_c w = μ (Σ_rest + ε I) w,       ε = 1e-6,

and the six filters with the largest eigenvalues are kept. Features are the log-variances
of the filtered signals (n_bands × n_classes × 6 values), standardized; the k = min(24,
n_features − 1) features with the largest mutual information (random_state = 42) feed an
LDA classifier. The penultimate representation used for head fine-tuning is the output of
the selection step.

This is filter-bank CSP (Ang et al., Front. Neurosci. 2012) restricted to four bands.
"""
from functools import partial

import numpy as np
from scipy import linalg as sla
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..filterbank import SPECTRAL_CSP_BANDS, filter_bank_dict, normalize_channels, to_3d

N_COMPONENTS = 6
EPSILON = 1e-6
COV_REG = 1e-6
SELECT_K = 24
MI_SEED = 42


def compute_covariance(X, reg=COV_REG):
    """Average spatial covariance of trials X (n, ch, t), each trial demeaned in time."""
    n_trials, n_ch, _ = X.shape
    covs = []
    for i in range(n_trials):
        Xi = X[i]
        Xi = Xi - Xi.mean(axis=1, keepdims=True)
        covs.append(Xi @ Xi.T)
    cov = np.mean(covs, axis=0)
    cov += reg * np.eye(n_ch)
    return cov


def solve_csp(Sigma1, Sigma2, n_components=N_COMPONENTS, epsilon=EPSILON):
    """Filters of Σ1 w = μ (Σ2 + ε I) w with the ``n_components`` largest eigenvalues."""
    n_ch = Sigma1.shape[0]
    denom = Sigma2 + epsilon * np.eye(n_ch)
    denom = (denom + denom.T) / 2.0
    eigs = np.linalg.eigvalsh(denom)
    if eigs.min() <= 0:
        denom += (-eigs.min() + 1e-8) * np.eye(n_ch)
    mu, W = sla.eigh(Sigma1, denom)
    idx = np.argsort(mu)[::-1]
    return W[:, idx[:n_components]], mu[idx[:n_components]]


def fit_ovr_csp(X, y, n_components=N_COMPONENTS, epsilon=EPSILON, reg=COV_REG):
    """One-vs-rest CSP filters, {class: W (ch, n_components)}."""
    n_classes = int(np.max(y)) + 1
    covs = {}
    for c in range(n_classes):
        mask = (y == c)
        if mask.sum() > 0:
            covs[c] = compute_covariance(X[mask], reg=reg)
    filters = {}
    for c in range(n_classes):
        if c not in covs:
            continue
        rest_mask = (y != c)
        if rest_mask.sum() == 0:
            continue
        Sigma_rest = compute_covariance(X[rest_mask], reg=reg)
        W, _ = solve_csp(covs[c], Sigma_rest, n_components=n_components, epsilon=epsilon)
        filters[c] = W
    return filters


def log_variance_features(X, filters):
    """Log-variance of the CSP-filtered trials, (n, n_classes · n_components)."""
    W = np.concatenate([filters[c] for c in sorted(filters.keys())], axis=1)
    feats = np.zeros((X.shape[0], W.shape[1]))
    for i in range(X.shape[0]):
        var = np.var(W.T @ X[i], axis=1)
        feats[i] = np.log(np.clip(var, a_min=1e-12, a_max=None))
    return feats


def fit_filter_bank(bands_X, y, n_components=N_COMPONENTS, epsilon=EPSILON, reg=COV_REG):
    return {band: fit_ovr_csp(X, y, n_components=n_components, epsilon=epsilon, reg=reg)
            for band, X in bands_X.items()}


def filter_bank_features(bands_X, fb_filters):
    """Features of all bands, concatenated in sorted band-name order."""
    return np.concatenate([log_variance_features(bands_X[b], fb_filters[b])
                           for b in sorted(bands_X.keys())], axis=1)


def fit_fold(X_tr, y_tr, X_calib, y_calib, X_eval, y_eval):
    """Fit on the source trials of one fold; return features and probabilities.

    X_* are EA-aligned (n, 1, ch, t) arrays. The validation split of the fold is not used
    (closed-form fit, no early stopping).
    """
    X_tr, X_calib, X_eval = normalize_channels(X_tr, X_calib, X_eval)
    X_tr, X_calib, X_eval = to_3d(X_tr), to_3d(X_calib), to_3d(X_eval)

    fb_tr = filter_bank_dict(X_tr, SPECTRAL_CSP_BANDS)
    fb_calib = filter_bank_dict(X_calib, SPECTRAL_CSP_BANDS)
    fb_eval = filter_bank_dict(X_eval, SPECTRAL_CSP_BANDS)

    fb_filters = fit_filter_bank(fb_tr, y_tr)
    f_tr = filter_bank_features(fb_tr, fb_filters)
    f_calib = filter_bank_features(fb_calib, fb_filters)
    f_eval = filter_bank_features(fb_eval, fb_filters)

    k = min(SELECT_K, f_tr.shape[-1] - 1)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('select', SelectKBest(score_func=partial(mutual_info_classif, random_state=MI_SEED),
                               k=k)),
        ('classifier', LDA()),
    ])
    pipeline.fit(f_tr, y_tr)
    pre = Pipeline(pipeline.steps[:-1])
    return {
        'train_feats': pre.transform(f_tr).astype(np.float32), 'train_y': y_tr.astype(np.int64),
        'calib_feats': pre.transform(f_calib).astype(np.float32),
        'calib_y': y_calib.astype(np.int64),
        'eval_feats': pre.transform(f_eval).astype(np.float32), 'eval_y': y_eval.astype(np.int64),
        'train_proba_rerun': pipeline.predict_proba(f_tr).astype(np.float32),
        'calib_proba_rerun': pipeline.predict_proba(f_calib).astype(np.float32),
        'eval_proba_rerun': pipeline.predict_proba(f_eval).astype(np.float32),
        'pipeline': pipeline,
    }
