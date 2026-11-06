# -*- coding: utf-8 -*-
"""Within-subject baselines trained only on the target's labeled calibration trials.

For every target subject and budget N (trials per class, the same grids and the same
subsets as head fine-tuning: ``default_rng(1000 + fold)``), a decoder is trained from scratch
on the N labeled calibration trials per class and scored on the evaluation block. No data of
other subjects and no EA are used (a per-subject invertible spatial transform does not change
these within-subject decoders).

  csp_lda  per-channel z-score (subset statistics) → CSP (4 components, OAS, log power)
           → shrinkage LDA
  fbcsp    per-channel z-score (subset statistics) → the FBCSP pipeline of the backbones
           (nine 4-Hz bands, CSP 6 components, MI selection k = 23, LDA)
  eegnet   EEGNet-8,2 (dropout 0.25, kernel 32) on the raw band-passed signals; the subset is
           split 80/20 per class (``default_rng(2000 + fold)``) for early stopping (Adam 1e-3,
           batch 64, ≤ 500 epochs, patience 75 on validation accuracy)

The EEGNet numbers of the paper came from runs in which several datasets shared one random
stream (``torch.manual_seed(42)`` once per process); this script reseeds per dataset, so its
EEGNet values match the paper up to training noise. CSP+LDA and FBCSP are deterministic.

Output: results/analysis/within_subject_baselines.json
  {dataset: {method: {n_subjects, n_classes, chance, K_grid: {K5: {...}, ...}}}}

  python scripts/02_within_subject_baselines.py --datasets 2a cho2017 physionet openbmi
"""
import argparse
import copy
import os
import sys
import time
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mne
import numpy as np
import torch
import torch.nn as nn
from mne.decoding import CSP
from moabb.pipelines import FilterBank
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from calibration_budget import config
from calibration_budget.backbones.eegnet import EEGNet
from calibration_budget.backbones.fbcsp import CSP_COMPONENTS, SELECT_K, CSPWithLogVar
from calibration_budget.data import load_subjects
from calibration_budget.filterbank import NINE_BANDS, filter_bank_stack
from calibration_budget.head import draw_subset
from calibration_budget.training import make_loader

mne.set_log_level('WARNING')

OUT = 'within_subject_baselines.json'
METHODS = ['csp_lda', 'fbcsp', 'eegnet']
VAL_SEED_BASE = 2000


def csp_lda(X_cal, y_cal, X_eval, y_eval):
    Xc, Xe = X_cal[:, 0], X_eval[:, 0]
    eps = 1e-8
    mu = Xc.mean(axis=(0, 2), keepdims=True)
    sd = Xc.std(axis=(0, 2), keepdims=True)
    Xc = (Xc - mu) / (sd + eps)
    Xe = (Xe - mu) / (sd + eps)
    csp = CSP(n_components=4, log=True, reg='oas')
    try:
        Fc = csp.fit_transform(Xc, y_cal)
        Fe = csp.transform(Xe)
    except Exception:                      # singular covariance
        return np.nan
    clf = LDA(solver='lsqr', shrinkage='auto')
    clf.fit(Fc, y_cal)
    return float((clf.predict(Fe) == y_eval).mean())


def fbcsp(X_cal, y_cal, X_eval, y_eval):
    Xs3, Xe3 = X_cal[:, 0].copy(), X_eval[:, 0].copy()
    eps = 1e-8
    for ch in range(Xs3.shape[1]):
        mu = float(Xs3[:, ch, :].mean())
        sd = float(Xs3[:, ch, :].std())
        Xs3[:, ch, :] = (Xs3[:, ch, :] - mu) / (sd + eps)
        Xe3[:, ch, :] = (Xe3[:, ch, :] - mu) / (sd + eps)
    fb_tr = filter_bank_stack(Xs3, NINE_BANDS)
    fb_ev = filter_bank_stack(Xe3, NINE_BANDS)
    k = min(SELECT_K, fb_tr.shape[-1] * CSP_COMPONENTS * 2 - 1)
    pipe = Pipeline([
        ('fbcsp', FilterBank(estimator=CSPWithLogVar(n_components=CSP_COMPONENTS, reg='oas',
                                                     log=True, norm_trace=False), flatten=True)),
        ('scaler', StandardScaler()),
        ('select', SelectKBest(score_func=partial(mutual_info_classif, random_state=42), k=k)),
        ('classifier', LDA()),
    ])
    try:
        pipe.fit(fb_tr, y_cal)
        return float((pipe.predict(fb_ev) == y_eval).mean())
    except Exception:
        return np.nan


def split_train_val(X, y, fold):
    rng = np.random.default_rng(VAL_SEED_BASE + fold)
    tr, va = np.zeros(len(y), dtype=bool), np.zeros(len(y), dtype=bool)
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        nv = max(1, int(round(0.2 * len(idx))))
        va[idx[:nv]] = True
        tr[idx[nv:]] = True
    return X[tr], y[tr], X[va], y[va]


def eegnet(X_tr, y_tr, X_val, y_val, X_te, y_te, n_classes, device,
           epochs=500, batch_size=64, lr=1e-3, patience=75):
    train_loader = make_loader(X_tr, y_tr, shuffle=True, batch_size=batch_size)
    val_loader = make_loader(X_val, y_val, shuffle=False, batch_size=batch_size)
    test_loader = make_loader(X_te, y_te, shuffle=False, batch_size=batch_size)
    model = EEGNet(n_classes=n_classes, channels=X_tr.shape[2], samples=X_tr.shape[3],
                   F1=8, D=2, kernel_len=32, dropout=0.25).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_val_acc, best_state, no_improve = 0.0, None, 0
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()
        model.eval()
        correct = n = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                correct += (model(xb).argmax(1) == yb).sum().item()
                n += xb.size(0)
        val_acc = correct / n
        if val_acc > best_val_acc:
            best_val_acc, best_state, no_improve = val_acc, copy.deepcopy(model.state_dict()), 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    correct = n = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            correct += (model(xb).argmax(1) == yb).sum().item()
            n += xb.size(0)
    return correct / n


def run(dataset, method, subjects):
    n_classes = config.N_CLASSES[dataset]
    sids = sorted(subjects)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if method == 'eegnet':
        torch.manual_seed(config.SEED)
    out = {'n_subjects': len(sids), 'n_classes': n_classes, 'chance': 1.0 / n_classes,
           'K_grid': {}}
    t0 = time.time()
    for K in config.BUDGETS[dataset]:
        accs = []
        for fold, sid in enumerate(sids):
            Xc, yc = subjects[sid]['X_train'], subjects[sid]['y_train']
            Xe, ye = subjects[sid]['X_test'], subjects[sid]['y_test']
            mask = draw_subset(yc, K, config.SUBSET_SEED_BASE + fold)
            if method == 'csp_lda':
                acc = csp_lda(Xc[mask], yc[mask], Xe, ye)
            elif method == 'fbcsp':
                acc = fbcsp(Xc[mask], yc[mask], Xe, ye)
            else:
                Xtr, ytr, Xva, yva = split_train_val(Xc[mask], yc[mask], fold)
                try:
                    acc = eegnet(Xtr, ytr, Xva, yva, Xe, ye, n_classes, device)
                except Exception as exc:          # e.g. validation accuracy never above 0
                    print(f'  [{dataset}|eegnet] K={K} fold {fold} failed: {exc}', flush=True)
                    acc = np.nan
            accs.append(acc)
        accs = np.array(accs, float)
        out['K_grid'][f'K{K}'] = {
            'total_trials': int(K * n_classes),
            'per_subject_acc': [None if np.isnan(a) else float(a) for a in accs],
            'mean': float(np.nanmean(accs)), 'std': float(np.nanstd(accs)),
            'n_valid_folds': int(np.sum(~np.isnan(accs)))}
        print(f'[{dataset}|{method}] K={K:<3} mean={out["K_grid"][f"K{K}"]["mean"]:.4f}  '
              f'({time.time() - t0:.0f} s)', flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--datasets', nargs='+', default=config.DATASETS, choices=config.DATASETS)
    ap.add_argument('--methods', nargs='+', default=METHODS, choices=METHODS)
    a = ap.parse_args()
    path = config.analysis_file(OUT)
    res = config.load_json(OUT) if os.path.exists(path) else {}
    for ds in a.datasets:
        subjects = load_subjects(ds)                # 4–40 Hz, no EA
        for m in a.methods:
            res.setdefault(ds, {})[m] = run(ds, m, subjects)
            config.save_json(OUT, res)
    print(f'[saved] {path}')


if __name__ == '__main__':
    main()
