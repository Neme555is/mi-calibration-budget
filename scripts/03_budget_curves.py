# -*- coding: utf-8 -*-
"""Budget curves of classifier-head fine-tuning and the shared-exponent power law.

For every backbone × dataset unit, the head is fine-tuned on N labeled calibration trials per
class (the dataset's three official budgets) and the accuracies are averaged over folds. The
curves are fitted with acc_e(N) = a_e − b_e·N^(−c) (shared c, free a_e and b_e).

Reported: pooled and per-dataset exponent and R², unit-level cluster bootstrap of c (2000
draws), per-unit parameters and residuals, leave-one-unit-out stability, the
identifiability of a per-unit exponent from three budgets, closed-form crossovers
N* = (b_e/(a_e − src_e))^(1/c), the shared-b model check, the within-subject baselines at
equal budgets, the frozen source models against the best within-subject baseline, and the
fit of the source models on their own training data.

Outputs (results/analysis/)
  budget_law.json       all four datasets (48 units; the paper's main fit)
  budget_law_3ds.json   with --datasets 2a cho2017 physionet (36 units; input of the
                        OpenBMI preregistration, scripts/09_preregistration.py)

  python scripts/03_budget_curves.py
  python scripts/03_budget_curves.py --datasets 2a cho2017 physionet --output budget_law_3ds.json
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from calibration_budget import config
from calibration_budget.curves import CurveSet, fit_shared_c, r2
from calibration_budget.head import fold_curves

WITHIN = 'within_subject_baselines.json'


def load_curves(datasets):
    """Mean budget curve and mean frozen-source accuracy of every unit."""
    curves, src = {}, {}
    for ds in datasets:
        Ks = config.BUDGETS[ds]
        for bk in config.BACKBONES:
            A, S = fold_curves(ds, bk, Ks)
            if len(A) == 0:
                raise FileNotFoundError(f'no features for {ds}|{bk}; run 01_train_source_models.py')
            curves[(ds, bk)] = {'N': list(Ks), 'acc': [float(np.mean(A[:, j])) for j in range(len(Ks))]}
            src[(ds, bk)] = float(np.mean(S))
            print(f'  [{ds}|{bk}] {len(S)} folds', flush=True)
    return curves, src


def source_fit(datasets):
    """Accuracy of the frozen source models on their training split, the target's
    calibration block and its evaluation block (an underfitted source model would inflate
    the headroom)."""
    out = {}
    for ds in datasets:
        out[ds] = {}
        for bk in config.BACKBONES:
            tr, ca, ev = [], [], []
            for fold in range(config.N_SUBJECTS[ds]):
                p = os.path.join(config.features_dir(ds), bk, f'fold{fold}.npz')
                if not os.path.exists(p):
                    continue
                z = np.load(p)
                for lst, split in ((tr, 'train'), (ca, 'calib'), (ev, 'eval')):
                    lst.append(float((z[f'{split}_proba_rerun'].argmax(1) == z[f'{split}_y']).mean()))
            out[ds][bk] = dict(n_folds=len(ev), train_acc=float(np.mean(tr)),
                               calib_acc=float(np.mean(ca)), eval_acc=float(np.mean(ev)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--datasets', nargs='+', default=config.DATASETS, choices=config.DATASETS)
    ap.add_argument('--output', default='budget_law.json')
    a = ap.parse_args()
    datasets = [d for d in config.DATASETS if d in a.datasets]

    curves, src = load_curves(datasets)
    cs = CurveSet(curves)
    out = {'labels': dict(datasets=config.DATASET_LABEL, backbones=config.BACKBONE_LABEL,
                          families=config.FAMILY_ORDER, family_of=config.FAMILY,
                          family_color=config.FAMILY_COLOR),
           'dataset_meta': {d: config.DATASET_META[d] for d in datasets},
           'budgets': {d: config.BUDGETS[d] for d in datasets}}

    # ── Pooled and per-dataset fits, unit-level cluster bootstrap ──────────────────────
    c_pool = fit_shared_c(cs)
    idx_of = {ds: np.array([i for i, k in enumerate(cs.keys) if k[0] == ds]) for ds in datasets}
    out['per_dataset'] = {ds: dict(c=fit_shared_c(cs, idx=idx_of[ds]),
                                   n_units=len(idx_of[ds])) for ds in datasets}
    for ds in datasets:
        out['per_dataset'][ds]['r2'] = r2(cs, out['per_dataset'][ds]['c'], idx_of[ds])
    rng = np.random.default_rng(42)
    n_units = len(cs.keys)
    boot = np.array([fit_shared_c(cs, idx=rng.integers(0, n_units, n_units)) for _ in range(2000)])
    out['pooled'] = dict(c=c_pool, r2=r2(cs, c_pool), n_units=n_units, n_points=int(cs.Y.size),
                         ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                         boot_c=[float(x) for x in boot])

    # ── Per-unit parameters and residuals under the pooled exponent ────────────────────
    a_, b_, _ = cs.fit(c_pool)
    pred = a_[:, None] - b_[:, None] * cs.N ** (-c_pool)
    resid_pp = (cs.Y - pred) * 100
    out['per_unit'] = {}
    for i, (ds, bk) in enumerate(cs.keys):
        out['per_unit'][f'{ds}|{bk}'] = dict(
            ds=ds, backbone=bk, label=config.BACKBONE_LABEL[bk], family=config.FAMILY[bk],
            src_acc=src[(ds, bk)], N=curves[(ds, bk)]['N'], acc=curves[(ds, bk)]['acc'],
            a=float(a_[i]), b=float(b_[i]),
            rmse_pp=float(np.sqrt(np.mean(resid_pp[i] ** 2))),
            max_resid_pp=float(np.max(np.abs(resid_pp[i]))),
            pred=[float(x) for x in pred[i]], resid_pp=[float(x) for x in resid_pp[i]])
    rms = np.sqrt((resid_pp ** 2).mean(1))
    out['residual_summary'] = dict(median=float(np.median(rms)), mean=float(rms.mean()),
                                   max=float(rms.max()), n_gt1=int((rms > 1).sum()),
                                   n_gt2=int((rms > 2).sum()), n=len(rms),
                                   rmse_values=[float(x) for x in rms])

    # Dataset means of source accuracy, headroom a − src and full-budget gain acc(N_max) − src.
    out['dataset_summary'] = {}
    for ds in datasets:
        u = [v for v in out['per_unit'].values() if v['ds'] == ds]
        out['dataset_summary'][ds] = dict(
            src_mean=float(np.mean([v['src_acc'] for v in u])),
            headroom_mean_pp=float(np.mean([v['a'] - v['src_acc'] for v in u])) * 100,
            gain_full_budget_mean_pp=float(np.mean([v['acc'][-1] - v['src_acc'] for v in u])) * 100)

    # ── Leave one unit out ─────────────────────────────────────────────────────────────
    c_v, err_v = [], []
    for i in range(n_units):
        idx = np.array([j for j in range(n_units) if j != i])
        c_i = fit_shared_c(cs, idx=idx)
        c_v.append(c_i)
        ai, bi, _ = cs.fit(c_i, idx=np.array([i]))
        p = ai[0] - bi[0] * cs.N[i] ** (-c_i)
        err_v.append(float(np.sqrt(np.mean(((cs.Y[i] - p) * 100) ** 2))))
    c_v, err_v = np.array(c_v), np.array(err_v)
    out['loo'] = dict(c_min=float(c_v.min()), c_max=float(c_v.max()), c_median=float(np.median(c_v)),
                      c_std=float(c_v.std()), c_std_rel_pct=float(100 * c_v.std() / c_pool),
                      holdout_median_pp=float(np.median(err_v)), holdout_mean_pp=float(err_v.mean()),
                      holdout_max_pp=float(err_v.max()), c_values=[float(x) for x in c_v],
                      holdout_errs=[float(x) for x in err_v])

    # ── Can a per-unit exponent be identified from three budgets? (0.5 pp noise, 200 draws) ──
    rng = np.random.default_rng(42)
    grid = np.arange(0.02, 1.51, 0.002)
    sigs = []
    for i in range(n_units):
        Yn = cs.Y[i][None, :] + rng.normal(0, 0.005, size=(200, cs.n))
        Syyn = ((Yn - Yn.mean(1, keepdims=True)) ** 2).sum(1)
        best = np.full(200, np.inf)
        arg = np.zeros(200)
        for c in grid:
            X = cs.N[i][None, :] ** (-c)
            Xc = X - X.mean(1, keepdims=True)
            Yc = Yn - Yn.mean(1, keepdims=True)
            Sxx = (Xc ** 2).sum(1)
            Sxy = (Xc * Yc).sum(1)
            s = np.maximum(Syyn - Sxy ** 2 / Sxx, 0.0)
            upd = s < best
            best = np.where(upd, s, best)
            arg = np.where(upd, c, arg)
        sigs.append(float(np.std(arg)))
    sigs = np.array(sigs)
    out['identifiability'] = dict(sigma_median=float(np.median(sigs)), sigma_mean=float(sigs.mean()),
                                  sigma_max=float(sigs.max()), pooled_c=c_pool,
                                  sigma_values=[float(x) for x in sigs], noise_pp=0.5, n_draws=200)

    # ── Closed-form crossover with the frozen source model ─────────────────────────────
    nstar = {}
    for i, (ds, bk) in enumerate(cs.keys):
        acc, s = cs.Y[i], src[(ds, bk)]
        rec = dict(head_at_max=float(acc[-1]), src=float(s), delta_max_pp=float(100 * (acc[-1] - s)))
        if a_[i] <= s:
            rec.update(N_star=None, reachable=False, note='asymptote not above the source level')
        else:
            n_ = float((b_[i] / (a_[i] - s)) ** (1.0 / c_pool))
            rec.update(N_star=n_, reachable=bool(n_ <= cs.N[i].max()), note='')
        nstar[f'{ds}|{bk}'] = rec
    out['per_unit_nstar'] = nstar

    # ── Collapsed coordinates (a − acc)/b ──────────────────────────────────────────────
    out['collapse'] = {f'{ds}|{bk}': dict(N=curves[(ds, bk)]['N'],
                                          y=[float((a_[i] - v) / b_[i]) for v in curves[(ds, bk)]['acc']])
                       for i, (ds, bk) in enumerate(cs.keys)}

    # ── Model check: free b_e versus one b shared by all units (budget scaled by N/D_e) ──
    D = np.array([[config.FEATURE_DIM[k[1]] for k in cs.keys]]).T

    def sse_shared_b(c):
        X = cs.N ** (-c) * D ** c
        Xc = X - X.mean(1, keepdims=True)
        Yc = cs.Y - cs.Ybar
        bsh = -float((Xc * Yc).sum()) / float((Xc ** 2).sum())
        A = cs.Y.mean(1) + bsh * X.mean(1)
        return float(((cs.Y - (A[:, None] - bsh * X)) ** 2).sum()), bsh

    g = np.arange(0.02, 1.51, 0.002)
    c0 = float(g[int(np.argmin([sse_shared_b(c)[0] for c in g]))])
    f = np.arange(max(0.01, c0 - 0.01), c0 + 0.01, 0.0001)
    c_sb = float(f[int(np.argmin([sse_shared_b(c)[0] for c in f]))])
    sse_sb, b_sb = sse_shared_b(c_sb)
    sse_free = cs.sse(c_pool)
    n_pts = int(cs.Y.size)
    df1, df2 = n_units - 1, n_pts - (2 * n_units + 1)
    F = ((sse_sb - sse_free) / df1) / (sse_free / df2)
    out['model_check'] = dict(c_free=c_pool, r2_free=r2(cs, c_pool), p_free=2 * n_units + 1,
                              df_free=df2, c_shared_b=c_sb, b_shared=float(b_sb),
                              r2_shared_b=1 - sse_sb / cs.sst(), p_shared_b=n_units + 2,
                              df_shared_b=n_pts - (n_units + 2), df1=df1, df2=df2, F=float(F),
                              n_points=n_pts, n_units=n_units)

    # ── Within-subject baselines at equal budgets, and the frozen source models ─────────
    wpath = config.analysis_file(WITHIN)
    ws = config.load_json(WITHIN) if os.path.exists(wpath) else {}
    front, k0 = {}, {}
    for ds in datasets:
        blk = ws.get(ds, {})
        if not all(m in blk for m in ('csp_lda', 'fbcsp', 'eegnet')):
            front[ds] = k0[ds] = dict(missing=True, note='run 02_within_subject_baselines.py')
            continue
        grid_ = blk['csp_lda']['K_grid']
        front[ds] = dict(K=[int(k[1:]) for k in grid_],
                         total=[int(grid_[k]['total_trials']) for k in grid_],
                         csp_lda=[float(grid_[k]['mean']) for k in grid_],
                         csp_lda_sd=[float(grid_[k]['std']) for k in grid_],
                         n_subj=int(blk['csp_lda']['n_subjects']),
                         chance=float(blk['csp_lda']['chance']))
        for m in ('fbcsp', 'eegnet'):
            front[ds][m] = [float(v['mean']) for v in blk[m]['K_grid'].values()]
        keys = [cs.keys[i] for i in idx_of[ds]]
        Ns = sorted({n for k in keys for n in curves[k]['N']})
        med, q1, q3 = [], [], []
        for n in Ns:
            vals = [curves[k]['acc'][curves[k]['N'].index(n)] for k in keys]
            med.append(float(np.median(vals)))
            q1.append(float(np.percentile(vals, 25)))
            q3.append(float(np.percentile(vals, 75)))
        front[ds]['units_median'] = dict(K=Ns, mean=med, q1=q1, q3=q3)
        front[ds]['src'] = float(np.mean([src[k] for k in keys]))

        names = ['csp_lda', 'fbcsp', 'eegnet']
        last = [front[ds][n][-1] for n in names]
        bi = int(np.argmax(last))
        best_val = float(last[bi])
        deltas = {f'{ds}|{k[1]}': float(100 * (src[k] - best_val)) for k in keys}
        k0[ds] = dict(best_within_name=names[bi], best_within=best_val,
                      within_total_trials=int(front[ds]['total'][-1]),
                      calib_cap=int(config.DATASET_META[ds]['calib_cap']), n_units=len(keys),
                      n_win=int(sum(1 for k in keys if src[k] > best_val)),
                      src_mean=float(np.mean([src[k] for k in keys])),
                      src_min=float(min(src[k] for k in keys)), src_max=float(max(src[k] for k in keys)),
                      delta_mean_pp=float(np.mean(list(deltas.values()))),
                      delta_best_pp=float(max(deltas.values())),
                      delta_worst_pp=float(min(deltas.values())), per_unit=deltas)
    out['within_subject'] = front
    out['source_vs_within_subject'] = k0
    out['source_fit'] = source_fit(datasets)

    path = config.save_json(a.output, out)
    print(f"[pooled ] c={c_pool:.4f}  R2={out['pooled']['r2']:.4f}  "
          f"CI95=[{out['pooled']['ci95'][0]:.3f}, {out['pooled']['ci95'][1]:.3f}]")
    print('[dataset] ' + '  '.join(f"{ds}: c={out['per_dataset'][ds]['c']:.4f} "
                                   f"R2={out['per_dataset'][ds]['r2']:.4f}" for ds in datasets))
    rs = out['residual_summary']
    print(f"[resid  ] median={rs['median']:.2f} mean={rs['mean']:.2f} max={rs['max']:.2f} pp  "
          f">1 pp: {rs['n_gt1']}/{rs['n']}")
    print(f"[LOO    ] c sd={c_v.std():.4f} ({out['loo']['c_std_rel_pct']:.1f}%)  "
          f"hold-out median={out['loo']['holdout_median_pp']:.2f} pp")
    mc = out['model_check']
    print(f"[model  ] free b R2={mc['r2_free']:.4f} | shared b R2={mc['r2_shared_b']:.4f}  "
          f"F({df1},{df2})={F:.2f}")
    for ds in datasets:
        k = k0[ds]
        if not k.get('missing'):
            print(f"[K=0    ] {ds:<10} source mean={k['src_mean']:.4f}  best within-subject="
                  f"{k['best_within_name']} {k['best_within']:.4f}  wins {k['n_win']}/{k['n_units']}")
    print(f'[saved  ] {path}')


if __name__ == '__main__':
    main()
