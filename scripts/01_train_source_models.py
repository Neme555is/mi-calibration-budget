# -*- coding: utf-8 -*-
"""Train (or fit) the twelve backbones leave-one-subject-out and save penultimate features.

For every fold k (target = k-th subject in sorted order) the backbone is trained on all
trials of the other subjects, and the features of the training split, the target's
calibration block and its evaluation block are saved together with the softmax outputs:

    results/features/<dataset>/<backbone>/fold{k}.npz
    results/checkpoints/<dataset>/<backbone>/fold{k}.pt     (.pkl for the CSP pipelines)

Random seeds (as in the paper)
------------------------------
* BCI IV-2a, Cho2017, PhysioNet: ``torch.manual_seed(seed)`` once per backbone before the
  first fold; later folds continue the same random stream. A run restricted to some folds,
  or resumed after an interruption, therefore starts those folds from a different state.
* OpenBMI: ``torch.manual_seed(seed + fold)`` before every fold, so any subset of folds
  reproduces a complete run. (211 of the 648 OpenBMI folds of the paper were produced with
  the per-backbone protocol before this change; see README.)
``--seed-mode`` overrides the default. GPU training is not bitwise deterministic, so reruns
agree with the paper up to run-to-run noise; the closed-form CSP pipelines are deterministic.

Examples
--------
  python scripts/01_train_source_models.py --dataset 2a
  python scripts/01_train_source_models.py --dataset physionet --backbones scsp fbcsp
  # EA switched off (front-end control of scripts/17_controls.py)
  python scripts/01_train_source_models.py --dataset 2a --no-ea --out-root results/controls/noea
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mne
import numpy as np
import torch

from calibration_budget import config
from calibration_budget.alignment import align_subjects
from calibration_budget.backbones import KIND
from calibration_budget.data import load_subjects
from calibration_budget.training import fit_fold, make_folds, save_checkpoint, save_features

mne.set_log_level('WARNING')

DEFAULT_SEED_MODE = {'2a': 'backbone', 'cho2017': 'backbone', 'physionet': 'backbone',
                     'openbmi': 'fold'}


def _seed(value):
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def run(dataset, backbones, folds, seed, seed_mode, apply_ea, out_root):
    feat_root = os.path.join(out_root, 'features', dataset)
    ckpt_root = os.path.join(out_root, 'checkpoints', dataset)
    n_classes = config.N_CLASSES[dataset]
    cache = {}
    for backbone in backbones:
        fmin, fmax = config.BAND[backbone]
        n_folds = config.N_SUBJECTS[dataset]
        todo = [k for k in range(n_folds) if (folds is None or k in folds)
                and not os.path.exists(os.path.join(feat_root, backbone, f'fold{k}.npz'))]
        if not todo:
            print(f'[{dataset}|{backbone}] all requested folds exist, skipped', flush=True)
            continue
        if (fmin, fmax) not in cache:
            cache.clear()
            t0 = time.perf_counter()
            subjects = load_subjects(dataset, fmin=fmin, fmax=fmax)
            cache[(fmin, fmax)] = align_subjects(subjects) if apply_ea else subjects
            print(f'[{dataset}] {len(subjects)} subjects, {fmin}-{fmax} Hz, '
                  f'EA={"on" if apply_ea else "off"}  ({time.perf_counter() - t0:.0f} s)',
                  flush=True)
        subjects = cache[(fmin, fmax)]
        print(f'[{dataset}|{backbone}] kind={KIND[backbone]} seed={seed} ({seed_mode}) '
              f'folds to run: {len(todo)}', flush=True)
        if seed_mode == 'backbone':
            _seed(seed)
        for fold, target, X_tr, y_tr, X_val, y_val, X_c, y_c, X_e, y_e in make_folds(subjects):
            if fold not in todo:
                continue
            if seed_mode == 'fold':
                _seed(seed + fold)
            t0 = time.perf_counter()
            out = fit_fold(backbone, n_classes, X_tr, y_tr, X_val, y_val, X_c, y_c, X_e, y_e)
            save_features(os.path.join(feat_root, backbone, f'fold{fold}.npz'), target, out)
            save_checkpoint(os.path.join(ckpt_root, backbone, f'fold{fold}'), out)
            acc = float((out['eval_proba_rerun'].argmax(1) == out['eval_y']).mean())
            print(f'  fold {fold:>3} subject {target:>3}  train={len(y_tr)} calib={len(y_c)} '
                  f'eval={len(y_e)}  source acc={acc:.4f}  ({time.perf_counter() - t0:.0f} s)',
                  flush=True)
            del out
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--dataset', required=True, choices=config.DATASETS)
    ap.add_argument('--backbones', nargs='+', default=config.BACKBONES, choices=config.BACKBONES)
    ap.add_argument('--folds', nargs='+', type=int, default=None)
    ap.add_argument('--seed', type=int, default=config.SEED)
    ap.add_argument('--seed-mode', choices=['backbone', 'fold'], default=None)
    ap.add_argument('--no-ea', action='store_true', help='switch Euclidean alignment off')
    ap.add_argument('--out-root', default=config.RESULTS_DIR,
                    help='features/ and checkpoints/ are written below this folder')
    a = ap.parse_args()
    run(a.dataset, a.backbones, set(a.folds) if a.folds else None, a.seed,
        a.seed_mode or DEFAULT_SEED_MODE[a.dataset], not a.no_ea, a.out_root)


if __name__ == '__main__':
    main()
