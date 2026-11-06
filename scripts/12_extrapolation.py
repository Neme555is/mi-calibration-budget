# -*- coding: utf-8 -*-
"""Can the budget law be carried to a new dataset? Leave-one-dataset-out extrapolation.

A  Identifiability of dataset-level covariates (class count, subject count, chance level,
   budget cap, mean source accuracy): rank of the centred design matrix over four datasets.
B  What the only unit-level covariate (source accuracy) carries: correlations of source
   accuracy, headroom and b over all units and within every dataset.
C  Leave-one-dataset-out prediction of the full-budget gain and of the headroom with a
   constant template, a source-accuracy template and a structured template
   gain(N) = headroom·(1 − k·N^(−c)).
D  Empirical prediction intervals from the out-of-sample errors of C.

Input: budget_law.json. Output: extrapolation.json

  python scripts/12_extrapolation.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from calibration_budget import config

ORDER = config.DATASETS


def load():
    D = config.load_json('budget_law.json')
    cells = {}
    for k, v in D['per_unit'].items():
        cells[k] = dict(ds=v['ds'], label=v['label'], src=float(v['src_acc']), a=float(v['a']),
                        b=float(v['b']), headroom=float(v['a'] - v['src_acc']),
                        gain_pp=float(100 * (v['acc'][-1] - v['src_acc'])), Kmax=int(v['N'][-1]))
    return cells, D['dataset_meta']


def identifiability(cells, meta):
    cols = {
        'intercept': [1.0] * len(ORDER),
        'n_classes': [meta[d]['n_class'] for d in ORDER],
        'n_subjects': [meta[d]['n_subj'] for d in ORDER],
        'chance': [meta[d]['chance'] for d in ORDER],
        'budget_cap_per_class': [meta[d]['calib_cap'] // meta[d]['n_class'] for d in ORDER],
        'mean_source_accuracy': [np.mean([c['src'] for c in cells.values() if c['ds'] == d]) for d in ORDER],
    }
    X = np.array(list(cols.values()), float).T
    s = np.linalg.svd(X - X.mean(0), full_matrices=False)[1]
    return dict(n_datasets=len(ORDER), n_covariates=len(cols), design_shape=list(X.shape),
                rank=int((s > 1e-9).sum()), covariate_names=list(cols.keys()))


def power(cells):
    ks = sorted(cells)
    col = lambda key: np.array([cells[k][key] for k in ks], float)
    src, head, b = col('src'), col('headroom'), col('b')
    ds = np.array([cells[k]['ds'] for k in ks])
    out = dict(n=len(ks), pooled=dict(src_headroom=float(np.corrcoef(src, head)[0, 1]),
                                      src_b=float(np.corrcoef(src, b)[0, 1]),
                                      headroom_b=float(np.corrcoef(head, b)[0, 1])), within={})
    for d in ORDER:
        m = ds == d
        out['within'][d] = dict(n=int(m.sum()), src_headroom=float(np.corrcoef(src[m], head[m])[0, 1]),
                                src_b=float(np.corrcoef(src[m], b[m])[0, 1]),
                                headroom_b=float(np.corrcoef(head[m], b[m])[0, 1]),
                                sd_headroom=float(head[m].std(ddof=1)), sd_src=float(src[m].std(ddof=1)))
    return out


def _fit_predict(tr, te, ykey, use_src):
    y = np.array([c[ykey] for c in tr], float)
    if not use_src:
        return float(y.mean()), float('nan')
    x = np.array([c['src'] for c in tr], float)
    A = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    ss = 1 - ((y - A @ coef) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return float(coef[0] + coef[1] * np.mean([c['src'] for c in te])), float(ss)


def _fit_structured(tr, te):
    """gain(N) = headroom·(1 − k·N^(−c)); (k, c) shared, headroom from the source template."""
    h = np.array([c['headroom'] for c in tr], float)
    N = np.array([c['Kmax'] for c in tr], float)
    g = np.array([c['gain_pp'] for c in tr], float) / 100.0
    C_HI = 0.80
    best = (np.inf, 0.24, 1.0)
    for cc in np.arange(0.02, C_HI, 0.002):
        u, v = h * N ** (-cc), h - g
        k = float(np.dot(u, v) / max(np.dot(u, u), 1e-12))
        sse = float(((u * k - v) ** 2).sum())
        if sse < best[0]:
            best = (sse, float(cc), k)
    _, c_hat, k_hat = best
    degenerate = bool(c_hat >= C_HI - 0.003 or k_hat > 20 or not np.isfinite(k_hat))
    xs = np.array([c['src'] for c in tr], float)
    A = np.vstack([np.ones_like(xs), xs]).T
    coef, *_ = np.linalg.lstsq(A, h, rcond=None)
    h_te = float(coef[0] + coef[1] * np.mean([c['src'] for c in te]))
    N_te = float(np.mean([c['Kmax'] for c in te]))
    return float(h_te * (1 - k_hat * N_te ** (-c_hat))), dict(
        c=c_hat, k=k_hat, headroom_pred=h_te, degenerate=degenerate, N_train=[float(x) for x in N])


def lodo(cells):
    res = {}
    for ykey in ('gain_pp', 'headroom'):
        folds = []
        for hold in ORDER:
            tr = [c for c in cells.values() if c['ds'] != hold]
            te = [c for c in cells.values() if c['ds'] == hold]
            obs = float(np.mean([c[ykey] for c in te]))
            row = dict(holdout=hold, observed=obs, n_test=len(te))
            for tag, use_src in (('const', False), ('src_linear', True)):
                pred, r2 = _fit_predict(tr, te, ykey, use_src)
                row[tag] = dict(pred=pred, err=pred - obs, abs_err=abs(pred - obs), train_r2=r2)
            if ykey == 'gain_pp':
                pred, info = _fit_structured(tr, te)
                obs_g = float(np.mean([c['gain_pp'] for c in te])) / 100.0
                row['structured'] = dict(pred=100 * pred, err=100 * (pred - obs_g),
                                         abs_err=100 * abs(pred - obs_g), **info)
            folds.append(row)
        errs = np.array([f['src_linear']['err'] for f in folds])
        errs0 = np.array([f['const']['err'] for f in folds])
        out = dict(folds=folds, mae_src_linear=float(np.abs(errs).mean()), mae_const=float(np.abs(errs0).mean()),
                   bias_src_linear=float(errs.mean()), rmse_src_linear=float(np.sqrt((errs ** 2).mean())))
        if ykey == 'gain_pp':
            ok = [f for f in folds if not f['structured']['degenerate']]
            e2 = np.array([f['structured']['err'] for f in ok])
            out['mae_structured'] = float(np.abs(e2).mean()) if len(e2) else float('nan')
            out['n_identifiable_folds'] = len(ok)
            out['degenerate_folds'] = [f['holdout'] for f in folds if f['structured']['degenerate']]
        res[ykey] = out
    return res


def intervals(cells, lodo_res):
    out = {}
    for ykey in ('gain_pp', 'headroom'):
        folds = lodo_res[ykey]['folds']
        errs = []
        for f in folds:
            for c in (c for c in cells.values() if c['ds'] == f['holdout']):
                errs.append((f['const']['pred'] - c[ykey], f['src_linear']['pred'] - c[ykey]))
        e0, e1 = np.array([e[0] for e in errs]), np.array([e[1] for e in errs])
        summ = lambda e: dict(mae=float(np.abs(e).mean()), sd=float(e.std(ddof=1)),
                              q025=float(np.percentile(e, 2.5)), q975=float(np.percentile(e, 97.5)))
        out[ykey] = dict(n_cell_errors=len(errs), const=summ(e0), src_linear=summ(e1),
                         fold_errors=[dict(holdout=f['holdout'], const_pp=float(f['const']['err']),
                                           src_linear_pp=float(f['src_linear']['err'])) for f in folds])
    return out


def main():
    cells, meta = load()
    lo = lodo(cells)
    res = dict(identifiability=identifiability(cells, meta), power=power(cells), lodo=lo,
               intervals=intervals(cells, lo), n_cells=len(cells))
    p = res['power']
    print(f"corr(src, headroom): pooled {p['pooled']['src_headroom']:+.3f}; within "
          + ', '.join(f"{d} {p['within'][d]['src_headroom']:+.3f}" for d in ORDER))
    print(f"LODO gain MAE: constant {lo['gain_pp']['mae_const']:.2f} pp, source template "
          f"{lo['gain_pp']['mae_src_linear']:.2f} pp, structured {lo['gain_pp']['mae_structured']:.2f} pp")
    print(f"[saved] {config.save_json('extrapolation.json', res)}")


if __name__ == '__main__':
    main()
