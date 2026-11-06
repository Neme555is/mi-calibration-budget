# -*- coding: utf-8 -*-
"""Sensitivity of the exponent to the budget grid (official versus dense).

The preregistration declared two OpenBMI grids before any OpenBMI model existed: the official
budgets [5, 20, 50] (used for the verdicts) and a dense grid [5, 10, 20, 35, 50]. The fold-level
accuracies on the dense grids of all datasets already exist, so no model is retrained.

1  P2b under both grids: the pooled exponent of 36 units (three datasets) and of 48 units,
   their difference, and a subject-level bootstrap (2000 draws) of the difference. The
   official grid must reproduce c_36, c_48 and their difference from the main analysis.
2  OpenBMI's own exponent on 3 versus 5 budgets, with subject-level bootstrap intervals, and
   whether each lies inside the preregistered P2a interval.

Inputs: foldlevel_curves.npz, crossover.json, preregistration_layerA.json.
Output: grid_density.json

  python scripts/11_grid_density.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from calibration_budget import config
from calibration_budget.curves import fit_c_boot, fit_c_grid_shared, pack

DS3 = ['2a', 'cho2017', 'physionet']
DS4 = DS3 + ['openbmi']
THRESHOLD = 0.03
N_BOOT = 2000
SEED = 20260917
GRID_COARSE = np.arange(0.02, 1.50, 0.002)


def fit_shared(Ns_list, Yb):
    """Shared exponent of every replicate of Yb (B, E, n) on ragged grids."""
    Yp, Np, W = pack(Ns_list, Yb)
    c0 = fit_c_boot(Yp, Np, W, GRID_COARSE)
    out = np.full(len(c0), np.nan)
    for b in range(len(c0)):
        if np.isfinite(c0[b]):
            fine = np.arange(max(0.01, c0[b] - 0.01), c0[b] + 0.01, 0.0001)
            out[b] = fit_c_boot(Yp[b:b + 1], Np, W, fine)[0]
    return out


def p2b_by_grid(z, emp):
    grids = {'official': emp['K_official'], 'dense': emp['K_fine']}
    keys_all = sorted(emp['K_official'])
    keys36 = [k for k in keys_all if k.split('|')[0] in DS3]
    n_subj = {ds: z[f'{ds}|{config.BACKBONES[0]}|A'].shape[0] for ds in DS4}
    M = {}
    for label, G in grids.items():
        for k in keys_all:
            allKs = np.asarray(z[f'{k}|Ks'], dtype=float)
            Ks = np.asarray(G[k], dtype=float)
            idx = [int(np.where(allKs == x)[0][0]) for x in Ks]
            M[(label, k)] = (Ks, z[f'{k}|A'][:, idx] * 100.0)

    def point(label, keys):
        Ns = [M[(label, k)][0] for k in keys]
        Y = np.zeros((1, len(keys), max(len(x) for x in Ns)))
        for i, k in enumerate(keys):
            Ks, Ym = M[(label, k)]
            Y[0, i, :len(Ks)] = Ym.mean(0)
        return Ns, Y

    rng = np.random.default_rng(SEED)
    picks = {ds: rng.integers(0, n_subj[ds], size=(N_BOOT, n_subj[ds])) for ds in DS4}

    def boot(label, keys):
        Ns = [M[(label, k)][0] for k in keys]
        Yb = np.zeros((N_BOOT, len(keys), max(len(x) for x in Ns)))
        for i, k in enumerate(keys):
            Ks, Y = M[(label, k)]
            Yb[:, i, :len(Ks)] = Y[picks[k.split('|')[0]]].mean(axis=1)
        return Ns, Yb

    out = {}
    for label in ('official', 'dense'):
        c36 = float(fit_shared(*point(label, keys36))[0])
        N48, Y48 = point(label, keys_all)
        c48 = float(fit_shared(N48, Y48)[0])
        db = fit_shared(*boot(label, keys_all)) - fit_shared(*boot(label, keys36))
        lo, hi = np.nanpercentile(db, [2.5, 97.5])
        out[label] = dict(c_36=c36, c_48=c48, delta=c48 - c36,
                          verdict='FAIL' if abs(c48 - c36) > THRESHOLD else 'PASS',
                          boot_median=float(np.nanmedian(db)), boot_ci95=[float(lo), float(hi)],
                          p_abs_delta_gt_threshold=float(np.nanmean(np.abs(db) > THRESHOLD)),
                          grid_len=sorted({len(x) for x in N48}))
        print(f'  [{label}] c_36 = {c36:.4f}  c_48 = {c48:.4f}  delta = {c48 - c36:+.4f}  '
              f'bootstrap CI [{lo:+.3f}, {hi:+.3f}]')
    return out


def openbmi_exponent(z, p2a):
    archs = sorted({k.split('|')[1] for k in z.files if k.startswith('openbmi|') and k.endswith('|A')})
    Ks = np.asarray(z[f'openbmi|{archs[0]}|Ks'], dtype=float)
    A = np.stack([z[f'openbmi|{a}|A'] for a in archs])            # (backbones, subjects, budgets)
    Y = A.mean(1)
    idx_off = [int(np.where(Ks == k)[0][0]) for k in config.BUDGETS['openbmi']]
    c3 = fit_c_grid_shared([Ks[idx_off]] * len(archs), Y[:, idx_off])
    c5 = fit_c_grid_shared([Ks] * len(archs), Y)
    rng = np.random.default_rng(SEED)
    n_subj = A.shape[1]

    def boot(cols):
        vals = np.empty(N_BOOT)
        for b in range(N_BOOT):
            pick = rng.integers(0, n_subj, n_subj)
            vals[b] = fit_c_grid_shared([Ks[cols]] * len(archs), A[:, pick][:, :, cols].mean(1))
        lo, hi = np.percentile(vals, [2.5, 97.5])
        return dict(median=float(np.median(vals)), ci95=[float(lo), float(hi)], width=float(hi - lo))

    b3 = boot(idx_off)
    b5 = boot(list(range(len(Ks))))
    lo, hi = p2a
    res = dict(grid_official=config.BUDGETS['openbmi'], grid_dense=Ks.tolist(), n_arch=len(archs),
               n_subj=int(n_subj), c_official_3pt=float(c3), c_dense_5pt=float(c5),
               span_pp_3pt=float((Y[:, idx_off].max(1) - Y[:, idx_off].min(1)).mean() * 100),
               span_pp_5pt=float((Y.max(1) - Y.min(1)).mean() * 100), boot_3pt=b3, boot_5pt=b5,
               prereg_P2a_interval=[lo, hi], P2a_inside_3pt=bool(lo <= c3 <= hi),
               P2a_inside_5pt=bool(lo <= c5 <= hi),
               P2a_ci_overlaps_3pt=bool(b3['ci95'][1] >= lo and b3['ci95'][0] <= hi),
               P2a_ci_overlaps_5pt=bool(b5['ci95'][1] >= lo and b5['ci95'][0] <= hi))
    print(f'  OpenBMI exponent: 3 budgets {c3:.4f} {b3["ci95"]}, 5 budgets {c5:.4f} {b5["ci95"]}; '
          f'P2a interval [{lo:.3f}, {hi:.3f}]')
    return res


def main():
    z = np.load(config.analysis_file('foldlevel_curves.npz'), allow_pickle=True)
    emp = config.load_json('crossover.json')
    p2a = config.load_json('preregistration_layerA.json')['predictions']['P2a']['interval']
    res = dict(p2b_by_grid=p2b_by_grid(z, emp), openbmi_exponent=openbmi_exponent(z, p2a))
    print(f"[saved] {config.save_json('grid_density.json', res)}")


if __name__ == '__main__':
    main()
