# -*- coding: utf-8 -*-
"""Euclidean alignment (EA; He & Wu, IEEE TBME 2020).

For each subject, the reference covariance R = (1/N) Σ_i X_i X_iᵀ is estimated from that
subject's own unlabeled trials and every trial is whitened, X̃_i = R^{-1/2} X_i, so that
the average spatial covariance of every subject becomes the identity.

Two estimation windows are used for the target subject:
  * ``full``  — R from all of its trials (calibration and evaluation blocks). This is the
    common transductive setting of offline evaluations.
  * ``calib`` — R from the calibration block only (strictly causal).
Source subjects always use all of their trials, in both windows.
"""
import numpy as np


def reference_inv_sqrt(X):
    """R^{-1/2} of the average spatial covariance of trials X with shape (N, C, T)."""
    n = X.shape[0]
    R = np.einsum('nct,ndt->cd', X, X) / n
    eigvals, eigvecs = np.linalg.eigh(R)
    eigvals = np.maximum(eigvals, 1e-10)
    return eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T


def apply_alignment(X, R_inv_sqrt):
    """Apply X̃ = R^{-1/2} X trial by trial; accepts (N, 1, C, T) or (N, C, T)."""
    squeeze = X.ndim == 4 and X.shape[1] == 1
    Xf = X[:, 0, :, :] if squeeze else X
    Xa = np.einsum('cd,ndt->nct', R_inv_sqrt, Xf).astype(np.float32)
    return Xa[:, None, :, :] if squeeze else Xa


def _as_3d(X):
    return X[:, 0, :, :] if (X.ndim == 4 and X.shape[1] == 1) else X


def align_subject(data, calibration_only=False):
    """Align one subject. ``data`` holds X_train/y_train (calibration block) and
    X_test/y_test (evaluation block); with ``calibration_only`` the reference covariance
    is estimated from the calibration block alone."""
    Xtr, Xte = data['X_train'], data['X_test']
    ref = Xtr if calibration_only else np.concatenate([Xtr, Xte], axis=0)
    R_inv_sqrt = reference_inv_sqrt(_as_3d(ref))
    return {'X_train': apply_alignment(Xtr, R_inv_sqrt), 'y_train': data['y_train'].copy(),
            'X_test': apply_alignment(Xte, R_inv_sqrt), 'y_test': data['y_test'].copy()}


def align_subjects(subjects, calibration_only_for=None):
    """Align every subject with its own reference covariance.

    ``subjects`` maps subject id → {X_train, y_train, X_test, y_test}. The subject whose id
    equals ``calibration_only_for`` (the target of a fold) uses the calibration-only window;
    all others use all of their trials. The input dictionary is not modified.
    """
    return {sid: align_subject(d, calibration_only=(sid == calibration_only_for))
            for sid, d in subjects.items()}
