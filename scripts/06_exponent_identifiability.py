# -*- coding: utf-8 -*-
"""Can each unit have its own exponent? A five-budget test on three datasets.

With three budgets per unit a free per-unit exponent c_e leaves no degree of freedom, so
"not identifiable" cannot be read as "equal". Here every unit is measured at five budgets
(2a [5, 10, 20, 36, 72], Cho2017 [5, 10, 20, 34, 67], PhysioNet [4, 6, 8, 11, 15]); the
features are reused, only the labeled subsets change. Reported per dataset:

  * shared c versus free c_e: R², parameters, degrees of freedom and the F statistic;
  * per-unit ĉ_e with bootstrap standard errors (subjects resampled, 200 draws);
  * variance components τ² = Var(ĉ_e) − mean(SE²) and a DerSimonian–Laird random-effects
    meta-analysis (Cochran Q, τ², I², pooled c).

Output: results/analysis/exponent_identifiability.json (the five-budget curves are also
used by scripts/07_functional_forms.py).

  python scripts/06_exponent_identifiability.py --dataset all
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy import stats

from calibration_budget import config
from calibration_budget.head import fold_curves

OUT = 'exponent_identifiability.json'
DATASETS = ['2a', 'cho2017', 'physionet']


def curves(ds, Ks):
    sub, per_fold = {}, {}
    for bk in config.BACKBONES:
        A, _ = fold_curves(ds, bk, Ks)
        key = f'{ds}|{bk}'
        sub[key] = [(K, float(np.mean(A[:, j]))) for j, K in enumerate(Ks)]
        per_fold[key] = {K: [float(x) for x in A[:, j]] for j, K in enumerate(Ks)}
    return sub, per_fold


def fit_ab_given_c(N, y, c):
    x = np.asarray(N, float) ** (-c)
    A = np.vstack([np.ones_like(x), -x]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef, float(np.sum((y - A @ coef) ** 2))


def fit_c_free(N, y, lo=0.01, hi=2.0):
    g = np.arange(lo, hi, 0.002)
    best, bc = np.inf, lo
    for c in g:
        _, s = fit_ab_given_c(N, y, c)
        if s < best:
            best, bc = s, c
    for c in np.arange(max(0.005, bc - 0.01), bc + 0.01, 0.0001):
        _, s = fit_ab_given_c(N, y, c)
        if s < best:
            best, bc = s, c
    coef, sse = fit_ab_given_c(N, y, bc)
    return float(bc), float(coef[0]), float(coef[1]), float(sse)


def fit_c_shared(sub, lo=0.01, hi=2.0):
    def sse(c):
        return sum(fit_ab_given_c([p[0] for p in pts], np.array([p[1] for p in pts]), c)[1]
                   for pts in sub.values())
    g = np.arange(lo, hi, 0.002)
    c0 = float(g[int(np.argmin([sse(c) for c in g]))])
    f = np.arange(max(0.005, c0 - 0.01), c0 + 0.01, 0.0001)
    return float(f[int(np.argmin([sse(c) for c in f]))]), sse


def analyse(sub):
    keys = sorted(sub.keys())
    E, n = len(keys), len(sub[keys[0]])
    n_pts = E * n
    c_shared, sse_fn = fit_c_shared(sub)
    sse_shared = sse_fn(c_shared)
    ce, sse_free = [], 0.0
    for k in keys:
        c, _, _, s = fit_c_free([p[0] for p in sub[k]], np.array([p[1] for p in sub[k]]))
        ce.append(c)
        sse_free += s
    ce = np.array(ce)
    par_shared, par_free = 2 * E + 1, 3 * E
    dof_shared, dof_free = n_pts - par_shared, n_pts - par_free
    df1 = par_free - par_shared
    F = ((sse_shared - sse_free) / df1) / (sse_free / dof_free) if dof_free > 0 else np.nan
    y_all = np.concatenate([np.array([p[1] for p in sub[k]]) for k in keys])
    sst = float(np.sum((y_all - y_all.mean()) ** 2))
    return dict(E=E, n_per_unit=n, n_pts=n_pts, c_shared=c_shared, r2_shared=1 - sse_shared / sst,
                ce=[float(x) for x in ce], ce_mean=float(ce.mean()), ce_std=float(ce.std()),
                r2_free=1 - sse_free / sst if sst > 0 else float('nan'),
                par_shared=par_shared, par_free=par_free, dof_shared=dof_shared,
                dof_free=dof_free, df1=df1, F=float(F), keys=keys)


def meta_analysis(ce, se):
    """Random-effects meta-analysis of the per-unit exponents (DerSimonian–Laird)."""
    ce, se = np.asarray(ce, float), np.asarray(se, float)
    ok = np.isfinite(ce) & np.isfinite(se) & (se > 0)
    ce, se = ce[ok], se[ok]
    k = len(ce)
    if k < 3:
        return dict(k=k, note='too few units')
    w = 1.0 / se ** 2
    c_fixed = float((w * ce).sum() / w.sum())
    Q = float((w * (ce - c_fixed) ** 2).sum())
    df = k - 1
    denom = float(w.sum() - (w ** 2).sum() / w.sum())
    tau2 = float(max(0.0, (Q - df) / denom)) if denom > 0 else 0.0
    ws = 1.0 / (se ** 2 + tau2)
    c_re = float((ws * ce).sum() / ws.sum())
    se_re = float(np.sqrt(1.0 / ws.sum()))
    shrink = ws / w
    return dict(k=k, c_fixed=c_fixed, Q=Q, df=df, p_Q=float(1 - stats.chi2.cdf(Q, df)),
                tau2=tau2, tau=float(np.sqrt(tau2)),
                I2_pct=float(max(0.0, (Q - df) / Q) * 100) if Q > 0 else 0.0,
                c_random=c_re, se_random=se_re, ci95_random=[c_re - 1.96 * se_re, c_re + 1.96 * se_re],
                shrink_mean=float(shrink.mean()), shrink_min=float(shrink.min()),
                shrink_max=float(shrink.max()), rejects_tau0=bool(1 - stats.chi2.cdf(Q, df) < 0.05))


def bootstrap_ce_se(per_fold, keys, Ks, n_boot=200, seed=42):
    """Bootstrap SE of every ĉ_e, resampling subjects."""
    rng = np.random.default_rng(seed)
    out = {}
    for k in keys:
        fk = per_fold[k]
        m = len(fk[Ks[0]])
        cs = []
        for _ in range(n_boot):
            idx = rng.integers(0, m, m)
            y = np.array([float(np.mean([fk[K][i] for i in idx])) for K in Ks])
            cs.append(fit_c_free(list(Ks), y)[0])
        out[k] = float(np.std(cs))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--dataset', default='all', choices=DATASETS + ['all'])
    ap.add_argument('--n-boot', type=int, default=200)
    a = ap.parse_args()
    path = config.analysis_file(OUT)
    res = config.load_json(OUT) if os.path.exists(path) else {}
    for ds in (DATASETS if a.dataset == 'all' else [a.dataset]):
        Ks = config.FIVE_POINT_BUDGETS[ds]
        t0 = time.perf_counter()
        sub, per_fold = curves(ds, Ks)
        r = analyse(sub)
        r['Ks'] = Ks
        r['curves'] = sub
        se = bootstrap_ce_se(per_fold, r['keys'], Ks, n_boot=a.n_boot)
        se_vals = np.array([se[k] for k in r['keys']])
        tau2 = max(r['ce_std'] ** 2 - float(np.mean(se_vals ** 2)), 0.0)
        r.update(se_ce={k: float(se[k]) for k in r['keys']}, tau2=float(tau2),
                 tau=float(np.sqrt(tau2)), mean_se=float(np.nanmean(se_vals)),
                 meta=meta_analysis(r['ce'], [se[k] for k in r['keys']]),
                 elapsed_min=round((time.perf_counter() - t0) / 60, 1))
        res[ds] = r
        config.save_json(OUT, res)
        ma = r['meta']
        print(f"{ds:<10} shared c={r['c_shared']:.4f} R2={r['r2_shared']:.4f} | free c_e R2={r['r2_free']:.4f} "
              f"F({r['df1']},{r['dof_free']})={r['F']:.2f} | tau={r['tau']:.4f}  "
              f"Q={ma['Q']:.2f} p={ma['p_Q']:.3f}  I2={ma['I2_pct']:.1f}%", flush=True)
    print(f'[saved] {path}')


if __name__ == '__main__':
    main()
