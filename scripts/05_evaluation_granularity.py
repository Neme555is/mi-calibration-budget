# -*- coding: utf-8 -*-
"""Evaluation granularity and statistical power of the budget curves.

A  Noise decomposition. The between-fold variance of the fold accuracies is split into a
   binomial part E[p(1−p)/n_eval] (set by the evaluation block size of each fold) and a
   between-subject part; the binomial share and the overdispersion are reported.
B  Minimum detectable difference of a dataset mean at 80 % power (two-sided α = 0.05):
   (t_.975 + t_.80) · SD_between / √n_subjects.
C  (--metrics) Continuous metrics: the same heads scored by AUC, log loss and Brier score,
   refitted with the shared exponent. Cached in results/analysis/foldlevel_metrics.npz.
D  Sampling standard error of AUC versus accuracy on the same evaluation trials
   (Hanley and McNeil, 1982).

Input: results/analysis/foldlevel_curves.npz (scripts/04_crossover.py).
Output: results/analysis/evaluation_granularity.json

  python scripts/05_evaluation_granularity.py --metrics
"""
import argparse
import collections
import io
import os
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.stats import t as tdist
from sklearn.metrics import log_loss, roc_auc_score

from calibration_budget import config
from calibration_budget.curves import CurveSet, fit_shared_c, r2
from calibration_budget.head import head_proba

CACHE = 'foldlevel_curves.npz'
MCACHE = 'foldlevel_metrics.npz'
OUT = 'evaluation_granularity.json'
EVAL_TRIALS = {'2a': 288, 'cho2017': 66, 'physionet': 15, 'openbmi': 100}
METRICS = ('acc', 'auc', 'logloss', 'brier')


def eval_sizes(ds):
    """Evaluation-block size of every fold (not constant on PhysioNet: 12–19 trials)."""
    out = []
    for fold in range(config.N_SUBJECTS[ds]):
        p = os.path.join(config.features_dir(ds), config.BACKBONES[0], f'fold{fold}.npz')
        if not os.path.exists(p):
            continue
        with zipfile.ZipFile(p) as z, z.open('eval_y.npy') as fh:
            out.append(int(np.load(io.BytesIO(fh.read())).shape[0]))
    return out


def decompose(z):
    out = {}
    for ds in config.DATASETS:
        Koff, Kfin = config.BUDGETS[ds], config.DENSE_BUDGETS[ds]
        idx = [Kfin.index(k) for k in Koff]
        keys = sorted(k[:-2] for k in z if k.endswith('|A') and k.startswith(ds + '|'))
        A = np.stack([z[k + '|A'][:, idx] for k in keys])            # (units, folds, budgets)
        E, nf, nK = A.shape
        nvec = np.array(eval_sizes(ds), float)
        if len(nvec) != nf:
            nvec = np.full(nf, float(EVAL_TRIALS[ds]))
        tot, bino = [], []
        for e in range(E):
            for j in range(nK):
                a = A[e, :, j]
                tot.append(float(a.var(ddof=1)))
                bino.append(float(np.mean(a * (1.0 - a) / nvec)))
        tot, bino = np.array(tot), np.array(bino)
        bet = np.maximum(tot - bino, 0.0)
        curve = A.mean((0, 1))
        se_point = float(np.sqrt(tot.mean() / nf))
        span = float(curve.max() - curve.min())
        out[ds] = dict(
            n_eval=EVAL_TRIALS[ds], n_eval_min=int(nvec.min()), n_eval_max=int(nvec.max()),
            n_subj=int(nf), K=[int(x) for x in Koff], curve=[float(x) for x in curve],
            span_pp=float(100 * span), sd_fold_pp=float(100 * np.sqrt(tot.mean())),
            sd_bino_pp=float(100 * np.sqrt(bino.mean())),
            sd_between_pp=float(100 * np.sqrt(bet.mean())),
            binom_share=float(bino.sum() / tot.sum()), phi=float(tot.mean() / bino.mean()),
            se_point_pp=float(100 * se_point),
            se_point_bino_pp=float(100 * np.sqrt(bino.mean() / nf)),
            se_point_between_pp=float(100 * np.sqrt(bet.mean() / nf)),
            span_over_se=float(span / se_point),
            sd_reduction_if_inf_eval=float(1.0 - np.sqrt(bet.mean() / tot.mean())))
    return out


def power_analysis(dec):
    out = {}
    for ds in config.DATASETS:
        n = dec[ds]['n_subj']
        sd_bet = dec[ds]['sd_between_pp'] / 100.0
        tcrit = tdist.ppf(0.975, n - 1) + tdist.ppf(0.80, n - 1)
        out[ds] = dict(n_subj=n, se_of_mean_pp=float(100 * sd_bet / np.sqrt(n)),
                       mde80_unpaired_pp=float(100 * tcrit * sd_bet / np.sqrt(n)))
    return out


def _metrics(proba, y):
    pred = proba.argmax(1)
    acc = float((pred == y).mean())
    cls = np.unique(y)
    p = np.clip(proba, 1e-12, 1.0)
    ll = float(log_loss(y, p / p.sum(1, keepdims=True), labels=cls))
    onehot = (y[:, None] == cls[None, :]).astype(float)
    brier = float(np.mean(np.sum((p / p.sum(1, keepdims=True) - onehot) ** 2, 1)))
    try:
        auc = (float(roc_auc_score(y, p[:, 1])) if len(cls) == 2
               else float(roc_auc_score(y, p, multi_class='ovr', average='macro')))
    except Exception:
        auc = float('nan')
    return acc, auc, ll, brier


def run_metrics(rebuild=False):
    path = config.analysis_file(MCACHE)
    if os.path.exists(path) and not rebuild:
        z = dict(np.load(path))
        return {k[:-4]: {m: z[f'{k[:-4]}|{m}'].tolist() for m in METRICS}
                for k in z if k.endswith('|acc')}
    res = {}
    for ds in config.DATASETS:
        Koff, nfold = config.BUDGETS[ds], config.N_SUBJECTS[ds]
        for bk in config.BACKBONES:
            kind = config.HEAD_KIND[bk]
            rows = {m: np.full((nfold, len(Koff)), np.nan) for m in METRICS}
            for fold in range(nfold):
                p = os.path.join(config.features_dir(ds), bk, f'fold{fold}.npz')
                if not os.path.exists(p):
                    continue
                f = np.load(p)
                Xc, yc = f['calib_feats'].astype(np.float64), f['calib_y'].astype(np.int64)
                Xe, ye = f['eval_feats'].astype(np.float64), f['eval_y'].astype(np.int64)
                for j, K in enumerate(Koff):
                    pr = head_proba(Xc, yc, Xe, K, config.SUBSET_SEED_BASE + fold, kind)
                    for m, v in zip(METRICS, _metrics(pr, ye)):
                        rows[m][fold, j] = v
            res[f'{ds}|{bk}'] = {m: v.tolist() for m, v in rows.items()}
            print(f'  [metrics] {ds}|{bk}', flush=True)
    np.savez_compressed(path, **{f'{k}|{m}': np.array(v[m]) for k, v in res.items() for m in METRICS})
    return res


def fit_c_from(blocks, kind):
    sub = {}
    for ds in config.DATASETS:
        for k, blk in blocks.items():
            if k.startswith(ds + '|'):
                sub[k] = {'N': config.BUDGETS[ds], 'acc': np.nanmean(np.array(blk[kind], float), axis=0).tolist()}
    cs = CurveSet(sub)
    c = fit_shared_c(cs)
    return float(c), float(r2(cs, c))


def auc_efficiency():
    """Sampling SE of AUC relative to accuracy on the same trials (Hanley–McNeil)."""
    out = {}
    for ds in config.DATASETS:
        cnts, accs, aucs = collections.Counter(), [], []
        for bk in config.BACKBONES:
            for fold in range(config.N_SUBJECTS[ds]):
                p = os.path.join(config.features_dir(ds), bk, f'fold{fold}.npz')
                if not os.path.exists(p):
                    continue
                f = np.load(p)
                ye = f['eval_y']
                cnts[tuple(sorted(collections.Counter(ye).values()))] += 1
                sp = f['eval_proba_rerun']
                accs.append(float((sp.argmax(1) == ye).mean()))
                if sp.shape[1] == 2:
                    aucs.append(float(roc_auc_score(ye, sp[:, 1])))
        k = cnts.most_common(1)[0][0]
        n = sum(k)
        pbar = float(np.mean(accs))
        se_acc = float(np.sqrt(pbar * (1 - pbar) / n))
        rec = dict(n_per_class=list(k), n_total=n, acc=pbar, se_acc=se_acc)
        if len(k) == 2 and aucs:
            n1, n2 = k
            A = float(np.mean(aucs))
            Q1, Q2 = A / (2 - A), 2 * A * A / (1 + A)
            se_auc = float(np.sqrt((A * (1 - A) + (n1 - 1) * (Q1 - A * A)
                                    + (n2 - 1) * (Q2 - A * A)) / (n1 * n2)))
            rec.update(auc=A, se_auc=se_auc, ratio=se_auc / se_acc)
        out[ds] = rec
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--metrics', action='store_true', help='also run part C (refits the heads)')
    a = ap.parse_args()
    z = dict(np.load(config.analysis_file(CACHE)))
    dec = decompose(z)
    res = dict(decompose=dec, power=power_analysis(dec), auc_efficiency=auc_efficiency())
    path = config.analysis_file(OUT)
    if os.path.exists(path):                       # keep part C of an earlier run
        for k, v in config.load_json(OUT).items():
            if k.startswith('metrics_') and k not in res:
                res[k] = v
    if a.metrics:
        mb = run_metrics()
        res['metrics_raw'] = mb
        res['metrics_fit'] = {kind: dict(zip(('c', 'r2'), fit_c_from(mb, kind))) for kind in METRICS}
        res['metrics_sd'] = {ds: {kind: float(np.mean([np.nanstd(np.array(mb[k][kind]), axis=0, ddof=1).mean()
                                                       for k in mb if k.startswith(ds + '|')]))
                                  for kind in METRICS} for ds in config.DATASETS}
    config.save_json(OUT, res)
    for ds in config.DATASETS:
        d, p = dec[ds], res['power'][ds]
        print(f'  {ds:<10} n_eval={d["n_eval"]:>3} subjects={d["n_subj"]:>3}  SD between={d["sd_between_pp"]:.2f} pp  '
              f'binomial share={100 * d["binom_share"]:.1f}%  span={d["span_pp"]:.2f} pp  '
              f'MDE80={p["mde80_unpaired_pp"]:.2f} pp')
    print(f'[saved] {path}')


if __name__ == '__main__':
    main()
