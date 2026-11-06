# -*- coding: utf-8 -*-
"""Leave-one-subject-out folds, source-model training and penultimate-feature extraction.

Protocol (identical for all gradient-trained backbones)
------------------------------------------------------
* Training pool of fold k: all trials (both blocks) of every subject except the target;
  10 % of the pooled trials, drawn with a NumPy ``RandomState(42)`` that advances from fold
  to fold, form the validation split used for early stopping.
* Per-channel z-scoring with the statistics of the training split, applied unchanged to the
  validation split and to the target's calibration and evaluation blocks.
* Adam (lr 1e-3), batch 64, cross-entropy, at most 500 epochs, early stopping with
  patience 75 on validation accuracy; the weights of the best validation epoch are kept.
  TSMNet uses Riemannian Adam (geoopt) on the CPU with the same schedule.
* The spectral CSP and FBCSP pipelines are fitted in closed form on the training split;
  they do not use the validation split.

Penultimate features are the inputs of the final classification layer, captured with a
forward pre-hook, for the training split, the calibration block and the evaluation block.
"""
import copy
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from . import backbones as B
from .backbones import fbcsp as fbcsp_pipeline
from .backbones import spectral_csp as scsp_pipeline
from .filterbank import fbcnet_input, normalize_bands, normalize_channels, to_3d

EPOCHS = 500
BATCH_SIZE = 64
LR = 1e-3
PATIENCE = 75
VAL_FRACTION = 0.1
SPLIT_SEED = 42


# ── Folds ─────────────────────────────────────────────────────────────────────
def _build_fold(subjects, all_subj, i, rng):
    target = all_subj[i]
    X_eval, y_eval = subjects[target]['X_test'], subjects[target]['y_test']
    X_calib, y_calib = subjects[target]['X_train'], subjects[target]['y_train']
    X_parts, y_parts = [], []
    for s in all_subj:
        if s == target:
            continue
        X_parts += [subjects[s]['X_train'], subjects[s]['X_test']]
        y_parts += [subjects[s]['y_train'], subjects[s]['y_test']]
    X_all = np.concatenate(X_parts, axis=0)
    y_all = np.concatenate(y_parts, axis=0)
    perm = rng.permutation(X_all.shape[0])
    n_val = max(1, int(VAL_FRACTION * X_all.shape[0]))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    return (i, target, X_all[train_idx], y_all[train_idx], X_all[val_idx], y_all[val_idx],
            X_calib, y_calib, X_eval, y_eval)


def make_folds(subjects):
    """Yield (fold, target, X_tr, y_tr, X_val, y_val, X_calib, y_calib, X_eval, y_eval).

    Folds follow the sorted subject ids. The validation split is drawn from one
    ``RandomState(42)`` stream that advances from fold to fold.
    """
    all_subj = sorted(subjects.keys())
    rng = np.random.RandomState(SPLIT_SEED)
    for i in range(len(all_subj)):
        yield _build_fold(subjects, all_subj, i, rng)


def get_fold(subjects, fold):
    """The tuple ``make_folds`` yields for one fold, without building the earlier folds.

    The split stream is advanced by the permutations of folds 0 … fold−1, which depend only
    on the training-pool sizes, so the result is identical to iterating ``make_folds``.
    """
    all_subj = sorted(subjects.keys())
    sizes = [len(subjects[s]['y_train']) + len(subjects[s]['y_test']) for s in all_subj]
    rng = np.random.RandomState(SPLIT_SEED)
    for i in range(fold):
        rng.permutation(sum(sizes) - sizes[i])
    return _build_fold(subjects, all_subj, fold, rng)


# ── Loaders and feature hooks ─────────────────────────────────────────────────
def make_loader(X, y, shuffle, batch_size=BATCH_SIZE):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y).long())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def _has_linear(m):
    if isinstance(m, nn.Linear):
        return True
    if isinstance(m, nn.Sequential):
        return any(isinstance(c, nn.Linear) for c in m)
    return False


def find_classifier(model):
    """The final classification layer (a Linear layer or a Sequential containing one)."""
    if isinstance(model, B.tsmnet.TSMNet):
        return model.classifier[0]
    for name in ('classifier', 'classification', 'fc', 'head', 'classification_head',
                 'lastLayer'):
        if hasattr(model, name) and _has_linear(getattr(model, name)):
            return getattr(model, name)
    cand = None
    for sub in model.modules():
        if sub is model:
            continue
        if isinstance(sub, nn.Linear):
            cand = sub
        elif isinstance(sub, nn.Sequential) and any(isinstance(c, nn.Linear) for c in sub):
            cand = sub
    if cand is None:
        raise RuntimeError('no classification layer found')
    return cand


def _forward(model, xb):
    if isinstance(model, B.tsmnet.TSMNet):
        return model(xb, d=None, return_latent=False)
    return model(xb)


def features_and_proba(model, loader, device):
    """Penultimate features (inputs of the classification layer) and softmax outputs."""
    clf = find_classifier(model)
    feats = []

    def pre_hook(_module, inp):
        x = inp[0]
        if x.dim() > 2:
            x = x.flatten(start_dim=1)
        feats.append(x.detach().cpu().clone())

    handle = clf.register_forward_pre_hook(pre_hook)
    model.eval()
    probas = []
    with torch.no_grad():
        for xb, _ in loader:
            probas.append(torch.softmax(_forward(model, xb.to(device)), dim=1).cpu().numpy())
    handle.remove()
    return (np.concatenate(feats, axis=0).astype(np.float32),
            np.concatenate(probas, axis=0).astype(np.float32))


# ── Input preparation per backbone ────────────────────────────────────────────
def prepare_inputs(backbone, X_tr, X_val, X_calib, X_eval):
    """Backbone-specific input transform and normalization (training-split statistics)."""
    kind = B.KIND[backbone]
    if kind == 'fbcnet':
        X_tr, X_val, X_calib, X_eval = map(fbcnet_input, (X_tr, X_val, X_calib, X_eval))
        return normalize_bands(X_tr, X_val, X_calib, X_eval)
    X_tr, X_val, X_calib, X_eval = normalize_channels(X_tr, X_val, X_calib, X_eval)
    if kind == 'tsmnet':
        X_tr, X_val, X_calib, X_eval = map(to_3d, (X_tr, X_val, X_calib, X_eval))
    return X_tr, X_val, X_calib, X_eval


def device_for(backbone):
    if B.KIND[backbone] == 'tsmnet':
        return torch.device('cpu')             # the SPD layers of TSMNet run on the CPU
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def _optimizer(backbone, model):
    if B.KIND[backbone] == 'tsmnet':
        from geoopt.optim import RiemannianAdam
        return RiemannianAdam(model.parameters(), lr=LR)
    return torch.optim.Adam(model.parameters(), lr=LR)


# ── Training ──────────────────────────────────────────────────────────────────
def train_network(backbone, n_classes, X_tr, y_tr, X_val, y_val, X_calib, y_calib,
                  X_eval, y_eval):
    """Train one gradient-based backbone on one fold; return features, probabilities and
    the best-validation weights."""
    X_tr, X_val, X_calib, X_eval = prepare_inputs(backbone, X_tr, X_val, X_calib, X_eval)
    device = device_for(backbone)
    tr = make_loader(X_tr, y_tr, shuffle=True)
    va = make_loader(X_val, y_val, shuffle=False)
    ca = make_loader(X_calib, y_calib, shuffle=False)
    ev = make_loader(X_eval, y_eval, shuffle=False)

    model = B.build(backbone, n_classes, X_tr).to(device)
    opt = _optimizer(backbone, model)
    crit = nn.CrossEntropyLoss()

    best, n_bad = 0.0, 0
    best_state = copy.deepcopy(model.state_dict())
    for _ in range(EPOCHS):
        model.train()
        for xb, yb in tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            crit(_forward(model, xb), yb).backward()
            opt.step()
        model.eval()
        correct = n = 0
        with torch.no_grad():
            for xb, yb in va:
                xb, yb = xb.to(device), yb.to(device)
                correct += (_forward(model, xb).argmax(1) == yb).sum().item()
                n += xb.size(0)
        val_acc = correct / n
        if val_acc > best:
            best, best_state, n_bad = val_acc, copy.deepcopy(model.state_dict()), 0
        else:
            n_bad += 1
        if n_bad >= PATIENCE:
            break

    model.load_state_dict(best_state)
    train_feats, train_proba = features_and_proba(
        model, make_loader(X_tr, y_tr, shuffle=False), device)
    calib_feats, calib_proba = features_and_proba(model, ca, device)
    eval_feats, eval_proba = features_and_proba(model, ev, device)
    return {
        'train_feats': train_feats, 'train_y': y_tr.astype(np.int64),
        'calib_feats': calib_feats, 'calib_y': y_calib.astype(np.int64),
        'eval_feats': eval_feats, 'eval_y': y_eval.astype(np.int64),
        'train_proba_rerun': train_proba, 'calib_proba_rerun': calib_proba,
        'eval_proba_rerun': eval_proba,
        'best_state': {k: v.cpu() for k, v in best_state.items()},
    }


def fit_fold(backbone, n_classes, X_tr, y_tr, X_val, y_val, X_calib, y_calib, X_eval, y_eval):
    """Train or fit ``backbone`` on one fold (EA-aligned, unnormalized (n, 1, ch, t) input)."""
    kind = B.KIND[backbone]
    if kind == 'scsp':
        return scsp_pipeline.fit_fold(X_tr, y_tr, X_calib, y_calib, X_eval, y_eval)
    if kind == 'fbcsp':
        return fbcsp_pipeline.fit_fold(X_tr, y_tr, X_calib, y_calib, X_eval, y_eval)
    return train_network(backbone, n_classes, X_tr, y_tr, X_val, y_val, X_calib, y_calib,
                         X_eval, y_eval)


def load_trained(backbone, n_classes, state, X_tr, device):
    """Rebuild a trained backbone from its saved weights (``X_tr`` = prepared input)."""
    model = B.build(backbone, n_classes, X_tr).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def forward_fold(backbone, n_classes, state, X_tr, X_val, X_calib, y_calib, X_eval, y_eval):
    """Features and probabilities of a trained backbone for the target blocks of one fold.

    ``X_tr`` / ``X_val`` are needed only for the normalization statistics and the input
    shape; nothing is trained.
    """
    X_tr, X_val, X_calib, X_eval = prepare_inputs(backbone, X_tr, X_val, X_calib, X_eval)
    device = device_for(backbone)
    model = load_trained(backbone, n_classes, state, X_tr, device)
    calib_feats, calib_proba = features_and_proba(
        model, make_loader(X_calib, y_calib, shuffle=False), device)
    eval_feats, eval_proba = features_and_proba(
        model, make_loader(X_eval, y_eval, shuffle=False), device)
    return {'calib_feats': calib_feats, 'calib_y': y_calib.astype(np.int64),
            'eval_feats': eval_feats, 'eval_y': y_eval.astype(np.int64),
            'calib_proba_rerun': calib_proba, 'eval_proba_rerun': eval_proba}


# ── Persistence ───────────────────────────────────────────────────────────────
FEATURE_KEYS = ('train_feats', 'train_y', 'calib_feats', 'calib_y', 'eval_feats', 'eval_y',
                'train_proba_rerun', 'calib_proba_rerun', 'eval_proba_rerun')


def save_features(path, test_subj, out):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, test_subj=test_subj, **{k: out[k] for k in FEATURE_KEYS if k in out})
    return path


def save_checkpoint(path_without_ext, out):
    """Atomically write the trained weights (.pt) or the fitted pipeline (.pkl)."""
    os.makedirs(os.path.dirname(path_without_ext), exist_ok=True)
    if 'best_state' in out:
        path = path_without_ext + '.pt'
        tmp = path + '.tmp'
        torch.save({'best_state': out['best_state']}, tmp)
    else:
        path = path_without_ext + '.pkl'
        tmp = path + '.tmp'
        with open(tmp, 'wb') as f:
            pickle.dump(out['pipeline'], f)
    os.replace(tmp, path)
    return path


def load_state(path_without_ext, device='cpu'):
    return torch.load(path_without_ext + '.pt', map_location=device,
                      weights_only=False)['best_state']
