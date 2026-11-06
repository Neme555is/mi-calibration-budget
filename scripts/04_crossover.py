# -*- coding: utf-8 -*-
"""Where fine-tuning overtakes the frozen source model, located on measured budgets only.

The closed-form crossover N* = (b_e/(a_e − src_e))^(1/c) depends on the fitted asymptote, which
is an extrapolation for most units. Here the crossover is located on the measured budgets
(interval censoring):

  L  the smallest budget already reaches the source level          N* ≤ K_min
  I  the curve crosses between two measured budgets                K_(j−1) < N* ≤ K_j
  R  the largest budget is still below the source level            N* > K_max

Denser budget grids are used for localization; each contains the dataset's official budgets,
from which the shared exponent is refitted. Uncertainty comes from a paired fold-level
bootstrap (2000 draws): folds are resampled with the accuracy matrix and the source level
together, and c, a_e, b_e are refitted in every replicate.

Also writes the fold-level accuracy matrices used by later scripts:
  results/analysis/foldlevel_curves.npz   keys '<ds>|<backbone>|A' (folds × dense budgets),
                                          '|S' (fold source accuracy), '|Ks' (dense budgets)
  results/analysis/crossover.json

  python scripts/04_crossover.py
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from calibration_budget import config
from calibration_budget.curves import crossing_from_A, fit_ab, fit_c_boot, fit_c_grid_shared, pack
from calibration_budget.head import fold_curves

CACHE = 'foldlevel_curves.npz'
OUT = 'crossover.json'
N_BOOT = 2000
BOOT_SEED = 20260911


def build_foldlevel(rebuild=False):
    """Fold-level matrices on the dense grids; cached and extended budget by budget."""
    path = config.analysis_file(CACHE)
    z = dict(np.load(path)) if (os.path.exists(path) and not rebuild) else {}
    store, n_new = {}, 0
    for ds in config.DATASETS:
        Kf = config.DENSE_BUDGETS[ds]
        for bk in config.BACKBONES:
            key = f'{ds}|{bk}'
            have_K = [int(x) for x in z[f'{key}|Ks']] if f'{key}|Ks' in z else []
            if f'{key}|A' in z and all(k in have_K for k in Kf):
                idx = [have_K.index(k) for k in Kf]
                store[key] = dict(A=np.asarray(z[f'{key}|A'], float)[:, idx],
                                  S=np.asarray(z[f'{key}|S'], float))
                continue
            t0 = time.perf_counter()
            missing = [k for k in Kf if k not in have_K]
            A_new, S = fold_curves(ds, bk, missing)
            A = np.zeros((len(S), len(Kf)))
            for j, k in enumerate(Kf):
                A[:, j] = (A_new[:, missing.index(k)] if k in missing
                           else np.asarray(z[f'{key}|A'], float)[:, have_K.index(k)])
            store[key] = dict(A=A, S=S)
            n_new += 1
            print(f'  [fold-level] {key:<26} {len(S):>3} folds × {len(Kf)} budgets '
                  f'({len(missing)} new)  {(time.perf_counter() - t0) / 60:.1f} min', flush=True)
    if n_new or not z:
        np.savez_compressed(path, **{f'{k}|A': v['A'] for k, v in store.items()},
                            **{f'{k}|S': v['S'] for k, v in store.items()},
                            **{f'{k}|Ks': np.array(config.DENSE_BUDGETS[k.split('|')[0]], int)
                               for k in store})
    return store


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--rebuild', action='store_true')
    ap.add_argument('--n-boot', type=int, default=N_BOOT)
    a = ap.parse_args()

    store = build_foldlevel(a.rebuild)
    DS = config.DATASETS
    keys = sorted(store, key=lambda k: (DS.index(k.split('|')[0]), k))
    E = len(keys)
    Koff = [config.BUDGETS[k.split('|')[0]] for k in keys]
    Kfin = [config.DENSE_BUDGETS[k.split('|')[0]] for k in keys]
    idx_off = [[Kfin[i].index(x) for x in Koff[i]] for i in range(E)]

    Yoff_mean = np.array([store[k]['A'][:, idx_off[i]].mean(0) for i, k in enumerate(keys)])
    Yfin_mean = [store[k]['A'].mean(0) for k in keys]
    S_mean = np.array([float(store[k]['S'].mean()) for k in keys])

    c_hat = fit_c_grid_shared(Koff, Yoff_mean)
    a_hat, b_hat = np.zeros(E), np.zeros(E)
    for i in range(E):
        aa, bb = fit_ab(Yoff_mean[i][None], Koff[i], c_hat)
        a_hat[i], b_hat[i] = float(aa[0]), float(bb[0])
    print(f'  shared c (official budgets) = {c_hat:.4f}', flush=True)

    per_cell = {}
    for i, k in enumerate(keys):
        ds, bk = k.split('|')
        acc, s = Yoff_mean[i], S_mean[i]
        Ns = np.array(Koff[i], float)
        cur = dict(ds=ds, backbone=bk, N_official=list(map(int, Koff[i])),
                   N_fine=list(map(int, Kfin[i])), src=float(s), acc=[float(x) for x in acc],
                   a=float(a_hat[i]), b=float(b_hat[i]),
                   delta_last_pp=float(100 * (acc[-1] - s)), delta_first_pp=float(100 * (acc[0] - s)))
        if a_hat[i] <= s:
            cur['N_model'] = None
            cur['N_model_note'] = 'a <= src: not reachable at any budget'
        else:
            cur['N_model'] = float((b_hat[i] / (a_hat[i] - s)) ** (1.0 / c_hat))
            cur['N_model_note'] = ''
        rise = float(acc[-1] - acc[0])
        cur['rho_unobserved'] = (float(a_hat[i] - acc[-1]) / rise) if rise > 1e-9 else float('inf')
        cur['extrap_x'] = (cur['N_model'] / Ns[-1]) if cur['N_model'] else None
        cur['emp_3pt'] = crossing_from_A(store[k]['A'][:, idx_off[i]], store[k]['S'], Koff[i])
        cur['emp_fine'] = crossing_from_A(store[k]['A'], store[k]['S'], Kfin[i])
        per_cell[k] = cur

    # ── Paired fold-level bootstrap ────────────────────────────────────────────────────
    print(f'  bootstrap B={a.n_boot} ...', flush=True)
    rng = np.random.default_rng(BOOT_SEED)
    nf = {k: store[k]['A'].shape[0] for k in keys}
    n_max_off = max(len(x) for x in Koff)
    n_max_fin = max(len(x) for x in Kfin)
    boot_idx = {k: rng.integers(0, nf[k], size=(a.n_boot, nf[k])) for k in keys}
    Yoff_b = np.zeros((a.n_boot, E, n_max_off))
    Yfin_b = np.zeros((a.n_boot, E, n_max_fin))
    S_b = np.zeros((a.n_boot, E))
    for i, k in enumerate(keys):
        A, S = store[k]['A'], store[k]['S']
        bi = boot_idx[k]
        Yoff_b[:, i, :len(Koff[i])] = A[:, idx_off[i]][bi].mean(1)
        Yfin_b[:, i, :len(Kfin[i])] = A[bi].mean(1)
        S_b[:, i] = S[bi].mean(1)
    Yp_off, Np_off, W_off = pack(Koff, Yoff_b, n_max_off)
    c_b = fit_c_boot(Yp_off, Np_off, W_off, np.arange(0.02, 1.50, 0.002))
    print(f'  c bootstrap: median {np.median(c_b):.4f}  '
          f'CI95 [{np.percentile(c_b, 2.5):.4f}, {np.percentile(c_b, 97.5):.4f}]', flush=True)

    Nmodel_b = np.full((a.n_boot, E), np.nan)
    for i in range(E):
        n = len(Koff[i])
        aa, bb = fit_ab(Yoff_b[:, i, :n], Koff[i], c_b)
        ok = aa > S_b[:, i]
        Nmodel_b[ok, i] = (bb[ok] / (aa[ok] - S_b[:, i][ok])) ** (1.0 / c_b[ok])

    for i, k in enumerate(keys):
        n = len(Kfin[i])
        Ns = np.array(Kfin[i], float)
        Y = Yfin_b[:, i, :n]
        Sb = S_b[:, i]
        hit = Y >= Sb[:, None]
        j = np.argmax(hit, axis=1)
        st = np.where(~hit.any(1), 2, np.where(j == 0, 0, 1))        # 0 = L, 1 = I, 2 = R
        nh = np.full(a.n_boot, np.nan)
        ii = st == 1
        if ii.any():
            jj = j[ii]
            a0, a1 = Y[ii, jj - 1], Y[ii, jj]
            n0, n1 = Ns[jj - 1], Ns[jj]
            fr = np.where(a1 > a0, (Sb[ii] - a0) / np.maximum(a1 - a0, 1e-12), 0.0)
            fr = np.clip(fr, 0, 1)
            nh[ii] = np.exp(np.log(n0) + fr * (np.log(n1) - np.log(n0)))
        cur = per_cell[k]
        fin = np.isfinite(nh).any()
        finm = np.isfinite(Nmodel_b[:, i]).any()
        cur['boot'] = dict(
            p_L=float((st == 0).mean()), p_I=float((st == 1).mean()), p_R=float((st == 2).mean()),
            N_emp_ci=[float(np.nanpercentile(nh, 2.5)), float(np.nanpercentile(nh, 97.5))] if fin else None,
            N_emp_median=float(np.nanmedian(nh)) if fin else None,
            p_model_defined=float(np.isfinite(Nmodel_b[:, i]).mean()),
            N_model_ci=[float(np.nanpercentile(Nmodel_b[:, i], 2.5)),
                        float(np.nanpercentile(Nmodel_b[:, i], 97.5))] if finm else None,
            N_model_median=float(np.nanmedian(Nmodel_b[:, i])) if finm else None,
            c_ci=[float(np.percentile(c_b, 2.5)), float(np.percentile(c_b, 97.5))])
        mc = cur['boot']['N_model_ci']
        cur['boot']['N_model_ci_decades'] = (float(np.log10(mc[1] / max(mc[0], 1e-9)))
                                             if mc and cur['N_model'] else None)
        ec = cur['boot']['N_emp_ci']
        cur['boot']['N_emp_ci_decades'] = float(np.log10(ec[1] / max(ec[0], 1e-9))) if ec else None

    per_ds = {}
    for ds in DS:
        ii = [i for i, k in enumerate(keys) if k.split('|')[0] == ds]
        Ns = np.array(config.DENSE_BUDGETS[ds], float)
        frac = [sum(1 for i in ii if (Yfin_mean[i][:j + 1] >= S_mean[i]).any()) / len(ii)
                for j in range(len(Ns))]
        los, his = [], []
        for n_idx in range(len(Ns)):
            fb = np.array([(Yfin_b[:, i, :n_idx + 1] >= S_b[:, i][:, None]).any(1) for i in ii]).mean(0)
            los.append(float(np.percentile(fb, 2.5)))
            his.append(float(np.percentile(fb, 97.5)))
        st = [per_cell[keys[i]]['emp_fine']['status'] for i in ii]
        per_ds[ds] = dict(N=list(map(int, Ns)), frac_crossed=frac, lo=los, hi=his, n_units=len(ii),
                          src_pooled=float(np.mean([S_mean[i] for i in ii])),
                          n_crossed_final=int(round(frac[-1] * len(ii))),
                          n_L=st.count('L'), n_I=st.count('I'), n_R=st.count('R'))

    res = dict(c_official=float(c_hat),
               c_boot_ci=[float(np.percentile(c_b, 2.5)), float(np.percentile(c_b, 97.5))],
               n_boot=a.n_boot, n_cells=E,
               K_official={k: list(map(int, Koff[i])) for i, k in enumerate(keys)},
               K_fine={k: list(map(int, Kfin[i])) for i, k in enumerate(keys)},
               per_cell=per_cell, per_dataset=per_ds)
    path = config.save_json(OUT, res)
    for ds in DS:
        d = per_ds[ds]
        print(f'  {ds:<10} L/I/R = {d["n_L"]}/{d["n_I"]}/{d["n_R"]}   '
              f'crossed at the largest budget: {d["frac_crossed"][-1]:.2f}')
    print(f'[saved] {path}')


if __name__ == '__main__':
    main()
