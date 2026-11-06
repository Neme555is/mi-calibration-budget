# -*- coding: utf-8 -*-
"""Classifier-head fine-tuning on frozen penultimate features.

For a target subject, N labeled calibration trials per class are drawn (stratified, without
replacement, ``numpy.random.default_rng(1000 + fold)``); a new classifier is fitted on their
standardized features and scored on the whole evaluation block. The classifier is L2
logistic regression whose C ∈ {0.01, 0.1, 1, 10} is chosen by stratified k-fold
cross-validation inside the labeled subset (k = 5 for N ≥ 20, otherwise 4), or
shrinkage LDA for the spectral CSP. The frozen source model itself (no labeled calibration)
is scored with its own softmax outputs (``eval_proba_rerun``).
"""
import os

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from . import config

C_GRID = [0.01, 0.1, 1.0, 10.0]


def draw_subset(y, N, seed):
    """Boolean mask of N trials per class (all trials of a class if it has fewer)."""
    rng = np.random.default_rng(seed)
    mask = np.zeros(len(y), dtype=bool)
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        mask[rng.choice(idx, size=min(N, len(idx)), replace=False)] = True
    return mask


def proba_head(X_tr, y_tr, X_q, C, kind):
    sc = StandardScaler().fit(X_tr)
    Xt, Xq = sc.transform(X_tr), sc.transform(X_q)
    if kind == 'lda_auto':
        clf = LDA(shrinkage='auto', solver='lsqr')
    else:
        clf = LogisticRegression(C=C, solver='lbfgs', max_iter=3000)
    clf.fit(Xt, y_tr)
    return clf.predict_proba(Xq)


def select_C(X_sub, y_sub, n_splits):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = {c: [] for c in C_GRID}
    for tr, te in skf.split(X_sub, y_sub):
        for c in C_GRID:
            try:
                p = proba_head(X_sub[tr], y_sub[tr], X_sub[te], c, 'lr')
                scores[c].append(float((p.argmax(1) == y_sub[te]).mean()))
            except Exception:
                scores[c].append(0.0)
    return max(C_GRID, key=lambda c: float(np.mean(scores[c])))


def head_proba(X_cal, y_cal, X_eval, N, seed, kind='lr'):
    """Posterior of the head fitted on N labeled calibration trials per class."""
    mask = draw_subset(y_cal, N, seed)
    Xs, ys = X_cal[mask], y_cal[mask]
    if kind == 'lda_auto':
        return proba_head(Xs, ys, X_eval, None, kind)
    return proba_head(Xs, ys, X_eval, select_C(Xs, ys, 5 if N >= 20 else 4), 'lr')


def head_at_N(X_cal, y_cal, X_eval, y_eval, N, seed, kind='lr'):
    """Evaluation accuracy of the head fitted on N labeled calibration trials per class."""
    p = head_proba(X_cal, y_cal, X_eval, N, seed, kind)
    return float((p.argmax(1) == y_eval).mean())


# ── Feature files ─────────────────────────────────────────────────────────────
def fold_path(dataset, backbone, fold, root=None):
    return os.path.join(root or config.features_dir(dataset), backbone, f'fold{fold}.npz')


def load_fold(path):
    """(X_calib, y_calib, X_eval, y_eval, eval_proba) of one saved fold."""
    f = np.load(path)
    return (f['calib_feats'].astype(np.float64), f['calib_y'].astype(np.int64),
            f['eval_feats'].astype(np.float64), f['eval_y'].astype(np.int64),
            f['eval_proba_rerun'].astype(np.float64))


def source_accuracy(eval_proba, y_eval):
    return float((eval_proba.argmax(1) == y_eval).mean())


def fold_curves(dataset, backbone, budgets, root=None, seed_base=config.SUBSET_SEED_BASE,
                n_folds=None):
    """Fold-level head accuracies A (n_folds, n_budgets) and source accuracies S (n_folds,)."""
    n_folds = n_folds or config.N_SUBJECTS[dataset]
    kind = config.HEAD_KIND[backbone]
    A, S = [], []
    for fold in range(n_folds):
        p = fold_path(dataset, backbone, fold, root)
        if not os.path.exists(p):
            continue
        Xc, yc, Xe, ye, sp = load_fold(p)
        A.append([head_at_N(Xc, yc, Xe, ye, K, seed_base + fold, kind) for K in budgets])
        S.append(source_accuracy(sp, ye))
    return np.array(A, float), np.array(S, float)
