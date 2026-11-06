# -*- coding: utf-8 -*-
"""Unsupervised domain adaptation against the budget law, on the same backbone and folds.

DANN, CDAN and CORAL are trained with EEGNet (the same network, EA front end, folds and
training schedule as the EEGNet backbone: Adam 1e-3, batch 64, ≤ 500 epochs, early stopping
with patience 75 on the source validation split). None of them uses target labels:

  native   the domain loss sees all target trials (calibration and evaluation blocks)
  matched  the domain loss sees 20 calibration trials per class of the target (unlabeled;
           drawn with ``default_rng(seed + 1000·fold + 20)``, with replacement when a class has
           fewer trials)

DANN and CDAN use a gradient-reversal layer with λ = 2/(1 + exp(−10p)) − 1 (p = epoch/epochs)
and a 64-unit domain discriminator on the 224-d penultimate features (CDAN: on the outer
product with the softmax output); CORAL adds ‖Cov_s − Cov_t‖²_F / D². Seeds 42, 43, 44; the
seed is set once per method before the first fold.

Comparison (three datasets, fold-wise, DA accuracy averaged over the three seeds; one-sided
Wilcoxon tests in the direction "DA is better", Holm correction over 18 cells):
  A  DA versus the frozen EEGNet source model (no labeled calibration)
  B  DA versus head fine-tuning of the same EEGNet with N labeled trials per class
     (N = 20 for 2a and Cho2017, 15 for PhysioNet)

Steps
  python scripts/16_domain_adaptation.py train --dataset 2a            (all methods, modes, seeds)
  python scripts/16_domain_adaptation.py compare
Outputs: results/domain_adaptation/<dataset>/<method>_<mode>_seed<s>/fold{k}.npz,
         results/analysis/domain_adaptation.json
"""
import argparse
import copy
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import wilcoxon
from torch.utils.data import DataLoader, TensorDataset

from calibration_budget import config
from calibration_budget.alignment import align_subjects
from calibration_budget.backbones.eegnet import EEGNet
from calibration_budget.data import load_subjects
from calibration_budget.filterbank import normalize_channels
from calibration_budget.head import head_at_N, load_fold
from calibration_budget.training import make_folds

DA_DIR = config.results_path('domain_adaptation')
DATASETS = ['2a', 'cho2017', 'physionet']
METHODS = ['dann', 'cdan', 'coral']
MODES = ['native', 'matched']
SEEDS = [42, 43, 44]
MATCHED_K = 20
N_HEAD = {'2a': 20, 'cho2017': 20, 'physionet': 15}
EPOCHS, BATCH_SIZE, LR, PATIENCE = 500, 64, 1e-3, 75
FEAT_DIM = 224


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x, lambd):
    return GradReverse.apply(x, lambd)


class DomainDiscriminator(nn.Module):
    def __init__(self, input_dim=FEAT_DIM, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2))

    def forward(self, x):
        return self.net(x)


class CDANDiscriminator(nn.Module):
    def __init__(self, feature_dim=FEAT_DIM, n_classes=4, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(feature_dim * n_classes, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, 2))

    def forward(self, x):
        return self.net(x)


def cdan_multilinear(feats, logits):
    sm = F.softmax(logits, dim=1)
    return torch.bmm(sm.unsqueeze(2), feats.unsqueeze(1)).view(feats.size(0), -1)


def coral_loss(feats_src, feats_tgt):
    D = feats_src.size(1)
    src_c = feats_src - feats_src.mean(dim=0, keepdim=True)
    tgt_c = feats_tgt - feats_tgt.mean(dim=0, keepdim=True)
    cov_src = src_c.t() @ src_c / max(feats_src.size(0) - 1, 1)
    cov_tgt = tgt_c.t() @ tgt_c / max(feats_tgt.size(0) - 1, 1)
    diff = cov_src - cov_tgt
    return (diff * diff).sum() / (D * D)


def kshot_sample(y, K, rng, n_classes):
    """K indices per class (with replacement only when a class has fewer than K trials)."""
    idx = []
    for c in range(n_classes):
        c_idx = np.where(y == c)[0]
        if len(c_idx) == 0:
            continue
        idx.append(rng.choice(c_idx, size=K, replace=len(c_idx) < K))
    return np.concatenate(idx) if idx else np.array([], dtype=int)


def _proba(model, X, y, device):
    loader = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y).long()),
                        batch_size=BATCH_SIZE, shuffle=False)
    model.eval()
    with torch.no_grad():
        return np.concatenate([torch.softmax(model(xb.to(device)), dim=1).cpu().numpy()
                               for xb, _ in loader]).astype(np.float64)


def da_fold(method, mode, fold, seed, n_classes, X_tr, y_tr, X_val, y_val, X_calib, y_calib, X_eval, y_eval):
    X_tr, X_val, X_calib, X_eval = normalize_channels(X_tr, X_val, X_calib, X_eval)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr).long()),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val).long()),
                            batch_size=BATCH_SIZE, shuffle=False)
    if mode == 'native':
        X_tgt_np = np.concatenate([X_calib, X_eval], axis=0)
    else:
        rng = np.random.default_rng(seed + fold * 1000 + MATCHED_K)
        X_tgt_np = X_calib[kshot_sample(y_calib, MATCHED_K, rng, n_classes)]
    X_tgt = torch.from_numpy(X_tgt_np).float().to(device)
    n_tgt = X_tgt.size(0)
    model = EEGNet(n_classes=n_classes, channels=X_tr.shape[2], samples=X_tr.shape[3],
                   F1=8, D=2, kernel_len=32, dropout=0.25).to(device)
    disc = None
    if method == 'dann':
        disc = DomainDiscriminator(input_dim=FEAT_DIM).to(device)
    elif method == 'cdan':
        disc = CDANDiscriminator(feature_dim=FEAT_DIM, n_classes=n_classes).to(device)
    params = list(model.parameters()) + (list(disc.parameters()) if disc is not None else [])
    optimizer = torch.optim.Adam(params, lr=LR)
    ce = nn.CrossEntropyLoss()
    best_val_acc, best_state, no_improve = 0.0, None, 0
    for epoch in range(EPOCHS):
        model.train()
        if disc is not None:
            disc.train()
        lambd = 2.0 / (1.0 + np.exp(-10.0 * epoch / max(EPOCHS - 1, 1))) - 1.0
        for xb_src, yb_src in train_loader:
            xb_src, yb_src = xb_src.to(device), yb_src.to(device)
            bs = xb_src.size(0)
            xb_tgt = X_tgt[torch.randint(0, n_tgt, (bs,), device=device)]
            logits_src, feats_src = model(xb_src, return_features=True)
            loss = ce(logits_src, yb_src)
            logits_tgt, feats_tgt = model(xb_tgt, return_features=True)
            if method in ('dann', 'cdan'):
                if method == 'dann':
                    d_src = disc(grad_reverse(feats_src, lambd))
                    d_tgt = disc(grad_reverse(feats_tgt, lambd))
                else:
                    d_src = disc(grad_reverse(cdan_multilinear(feats_src, logits_src), lambd))
                    d_tgt = disc(grad_reverse(cdan_multilinear(feats_tgt, logits_tgt), lambd))
                loss = (loss + ce(d_src, torch.zeros(bs, dtype=torch.long, device=device))
                        + ce(d_tgt, torch.ones(bs, dtype=torch.long, device=device)))
            else:
                loss = loss + 1.0 * coral_loss(feats_src, feats_tgt)
            optimizer.zero_grad()
            loss.backward()
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
        if no_improve >= PATIENCE:
            break
    model.load_state_dict(best_state)
    return _proba(model, X_calib, y_calib, device), _proba(model, X_eval, y_eval, device)


def train(ds, methods, modes, seeds):
    subjects = align_subjects(load_subjects(ds))
    n_classes = config.N_CLASSES[ds]
    for method in methods:
        for mode in modes:
            for seed in seeds:
                out_dir = os.path.join(DA_DIR, ds, f'{method}_{mode}_seed{seed}')
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                accs, t0 = [], time.perf_counter()
                for fold, target, X_tr, y_tr, X_val, y_val, X_c, y_c, X_e, y_e in make_folds(subjects):
                    calib_p, eval_p = da_fold(method, mode, fold, seed, n_classes,
                                              X_tr, y_tr, X_val, y_val, X_c, y_c, X_e, y_e)
                    os.makedirs(out_dir, exist_ok=True)
                    np.savez(os.path.join(out_dir, f'fold{fold}.npz'), test_subj=target,
                             calib_proba=calib_p.astype(np.float32), calib_y=y_c.astype(np.int64),
                             eval_proba=eval_p.astype(np.float32), eval_y=y_e.astype(np.int64))
                    accs.append(float((eval_p.argmax(1) == y_e).mean()))
                print(f'[{ds}] {method}_{mode} seed {seed}: mean accuracy {np.mean(accs):.4f} '
                      f'({(time.perf_counter() - t0) / 60:.1f} min)', flush=True)


def paired(da, ref, name):
    d = np.asarray(da, float) - np.asarray(ref, float)
    if not np.any(d != 0):
        p = p_rev = 1.0
    else:
        p = float(wilcoxon(d, alternative='greater').pvalue)
        p_rev = float(wilcoxon(d, alternative='less').pvalue)
    return dict(cell=name, n=int(len(d)), da=float(np.mean(da) * 100), ref=float(np.mean(ref) * 100),
                delta_pp=float(d.mean() * 100), wins=int((d > 0).sum()), p=p, p_rev=p_rev)


def holm(cells):
    order = sorted(range(len(cells)), key=lambda i: cells[i]['p'])
    m, prev = len(cells), 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, max(prev, (m - rank) * cells[i]['p']))
        cells[i]['p_holm'] = adj
        cells[i]['holm_survives'] = bool(adj < 0.05)
        prev = adj
    return cells


def compare():
    out = {'meta': dict(backbone='eegnet', n_head=N_HEAD, seeds=SEEDS, family=18,
                        direction='one-sided Wilcoxon, DA better')}
    per_ds, cells_A, cells_B = {}, [], []
    for ds in DATASETS:
        src, head, ye_all = [], [], []
        for f in range(config.N_SUBJECTS[ds]):
            Xc, yc, Xe, ye, sp = load_fold(os.path.join(config.features_dir(ds), 'eegnet', f'fold{f}.npz'))
            src.append(float((sp.argmax(1) == ye).mean()))
            head.append(head_at_N(Xc, yc, Xe, ye, N_HEAD[ds], config.SUBSET_SEED_BASE + f))
            ye_all.append(ye)
        src, head = np.array(src), np.array(head)
        d_sh = head - src
        rec = dict(src_pp=float(src.mean() * 100), head_pp=float(head.mean() * 100), N_head=N_HEAD[ds],
                   n_fold=len(src), cells=[], da_seed_spread={},
                   src_vs_head=dict(delta_pp=float(d_sh.mean() * 100), wins=int((d_sh > 0).sum()), n=len(d_sh),
                                    p=1.0 if not np.any(d_sh != 0) else
                                    float(wilcoxon(d_sh, alternative='greater').pvalue)))
        for m in METHODS:
            for mo in MODES:
                accs = []
                for s in SEEDS:
                    row = []
                    for f in range(config.N_SUBJECTS[ds]):
                        z = np.load(os.path.join(DA_DIR, ds, f'{m}_{mo}_seed{s}', f'fold{f}.npz'))
                        ye = z['eval_y'].astype(np.int64)
                        if not np.array_equal(ye, ye_all[f]):
                            raise RuntimeError(f'{ds}/{m}_{mo}_seed{s}/fold{f}: evaluation labels differ')
                        row.append(float((z['eval_proba'].argmax(1) == ye).mean()))
                    accs.append(row)
                accs = np.array(accs)
                da = accs.mean(0)
                rec['da_seed_spread'][f'{m}_{mo}_pp'] = float((accs.max(0) - accs.min(0)).mean() * 100)
                cA = paired(da, src, f'{ds}|{m}_{mo}|vs_K0')
                cB = paired(da, head, f'{ds}|{m}_{mo}|vs_head@N{N_HEAD[ds]}')
                rec['cells'] += [cA, cB]
                cells_A.append(cA)
                cells_B.append(cB)
        per_ds[ds] = rec
    holm(cells_A)
    holm(cells_B)
    out.update(per_dataset=per_ds, vs_K0=cells_A, vs_head=cells_B)
    for tag, cells in (('DA vs frozen source', cells_A), ('DA vs head fine-tuning', cells_B)):
        pos = sum(c['delta_pp'] > 0 for c in cells)
        print(f'  {tag}: {pos}/18 cells with Δ > 0, {sum(c["holm_survives"] for c in cells)}/18 survive Holm')
    print(f"[saved] {config.save_json('domain_adaptation.json', out)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('step', choices=['train', 'compare'])
    ap.add_argument('--dataset', choices=DATASETS)
    ap.add_argument('--methods', nargs='+', default=METHODS, choices=METHODS)
    ap.add_argument('--modes', nargs='+', default=MODES, choices=MODES)
    ap.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    a = ap.parse_args()
    if a.step == 'compare':
        compare()
        return
    if not a.dataset:
        ap.error('--dataset is required for training')
    train(a.dataset, a.methods, a.modes, a.seeds)


if __name__ == '__main__':
    main()
