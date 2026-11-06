# -*- coding: utf-8 -*-
"""Window of the EA reference covariance of the target subject: all trials versus calibration only.

  full   R of the target from its calibration and evaluation blocks (transductive; the
         main analysis)
  calib  R of the target from its calibration block only (causal)

Source subjects use all of their trials in both windows, so the trained source models are
reused unchanged: only the target's features are recomputed with the saved weights, and the
closed-form CSP pipelines are refitted on the identical training split.

Steps
  features  python scripts/13_ea_window.py features --dataset 2a --window calib
            → results/ea_window/<dataset>/<window>/<backbone>/fold{k}.npz
  curves    python scripts/13_ea_window.py curves --dataset 2a --window calib
            → results/analysis/ea_window_curves_<dataset>_<window>.json
              (fold-level head accuracies at the official budgets, source accuracies)
  report    python scripts/13_ea_window.py report
            → results/analysis/ea_window.json

The report gives, per dataset and window: source accuracy, full-budget gain Δ, backbones with
Δ > 0, between-subject SD, binomial share, curve span, minimum detectable difference and
span/MDE (with the main features as a reference column and paired Wilcoxon tests of the
per-backbone differences); P4b in both windows (frozen three-dataset exponent and refitted
exponent, 4-point permutation tests); the dataset headroom under five forms in both windows;
and the headroom ordering after removing DeepConvNet.
"""
import argparse
import itertools
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mne
import numpy as np
import torch
from scipy import stats

from calibration_budget import config
from calibration_budget.alignment import align_subject, align_subjects
from calibration_budget.backbones import KIND
from calibration_budget.curves import CurveSet, fit_shared_c
from calibration_budget.data import load_subjects
from calibration_budget.forms import FORMS, pooled_fit
from calibration_budget.head import head_at_N
from calibration_budget.summary import curve_summary
from calibration_budget.training import fit_fold, forward_fold, get_fold, load_state

mne.set_log_level('WARNING')

DS4 = config.DATASETS
WINDOWS = ['full', 'calib']
ORDER = ['2a', 'cho2017', 'physionet', 'openbmi']        # headroom ordering of the paper
PERMS = list(itertools.permutations(range(4)))
KEEP = ('calib_feats', 'calib_y', 'eval_feats', 'eval_y', 'eval_proba_rerun')


def window_dir(ds, window, backbone=''):
    return os.path.join(config.EA_WINDOW_DIR, ds, window, backbone)


def curves_file(ds, window):
    return f'ea_window_curves_{ds}_{window}.json'


# ── features ─────────────────────────────────────────────────────────────────────────
def features(ds, window, backbones, folds, validate):
    n_classes = config.N_CLASSES[ds]
    order = sorted(backbones, key=lambda b: (config.BAND[b], config.BACKBONES.index(b)))
    band = None
    for bk in order:
        if config.BAND[bk] != band:
            band = config.BAND[bk]
            raw = load_subjects(ds, fmin=band[0], fmax=band[1])
            aligned_full = align_subjects(raw)
        sids = sorted(raw)
        t0 = time.perf_counter()
        for i, target in enumerate(sids):
            if folds is not None and i not in folds:
                continue
            aligned = dict(aligned_full)
            if window == 'calib':
                aligned[target] = align_subject(raw[target], calibration_only=True)
            _, tsubj, X_tr, y_tr, X_val, y_val, X_c, y_c, X_e, y_e = get_fold(aligned, i)
            assert tsubj == target
            if KIND[bk] in ('scsp', 'fbcsp'):
                out = fit_fold(bk, n_classes, X_tr, y_tr, X_val, y_val, X_c, y_c, X_e, y_e)
            else:
                device = 'cpu' if KIND[bk] == 'tsmnet' else ('cuda' if torch.cuda.is_available() else 'cpu')
                state = load_state(os.path.join(config.CHECKPOINT_DIR, ds, bk, f'fold{i}'), device)
                out = forward_fold(bk, n_classes, state, X_tr, X_val, X_c, y_c, X_e, y_e)
            os.makedirs(window_dir(ds, window, bk), exist_ok=True)
            np.savez(os.path.join(window_dir(ds, window, bk), f'fold{i}.npz'), **{k: out[k] for k in KEEP})
            src = float((out['eval_proba_rerun'].argmax(1) == out['eval_y']).mean())
            msg = f'  [{ds}|{window}|{bk}] fold {i:>3} source acc={src:.4f}'
            if validate:
                ref = np.load(os.path.join(config.features_dir(ds), bk, f'fold{i}.npz'))
                src_ref = float((ref['eval_proba_rerun'].argmax(1) == ref['eval_y']).mean())
                msg += (f'  main={src_ref:.4f}  max|Δfeature|='
                        f'{np.abs(ref["eval_feats"].astype(float) - out["eval_feats"]).max():.2e}')
            print(msg, flush=True)
        print(f'[{ds}|{window}|{bk}] done in {time.perf_counter() - t0:.0f} s', flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ── curves ───────────────────────────────────────────────────────────────────────────
def curves(ds, window):
    Ks = config.BUDGETS[ds]
    cells = {}
    for bk in config.BACKBONES:
        A, S, nev = [], [], []
        for f in range(config.N_SUBJECTS[ds]):
            z = np.load(os.path.join(window_dir(ds, window, bk), f'fold{f}.npz'))
            Xc, yc = z['calib_feats'].astype(float), z['calib_y'].astype(int)
            Xe, ye = z['eval_feats'].astype(float), z['eval_y'].astype(int)
            A.append([head_at_N(Xc, yc, Xe, ye, K, config.SUBSET_SEED_BASE + f, config.HEAD_KIND[bk])
                      for K in Ks])
            S.append(float((z['eval_proba_rerun'].argmax(1) == ye).mean()))
            nev.append(len(ye))
        cells[bk] = dict(acc=A, src=S, n_eval=nev)
        print(f'  [{ds}|{window}] {bk}', flush=True)
    print(f"[saved] {config.save_json(curves_file(ds, window), dict(K=Ks, cells=cells))}")


# ── report ───────────────────────────────────────────────────────────────────────────
def _arms(ds):
    out = {}
    for w in WINDOWS:
        c = config.load_json(curves_file(ds, w))
        out[w] = (np.stack([np.array(c['cells'][bk]['acc'], float) for bk in config.BACKBONES]),
                  np.stack([np.array(c['cells'][bk]['src'], float) for bk in config.BACKBONES]),
                  np.array(c['cells'][config.BACKBONES[0]]['n_eval'], float))
    return out


def table(z):
    res = {}
    for ds in DS4:
        arms = _arms(ds)
        idx = [config.DENSE_BUDGETS[ds].index(k) for k in config.BUDGETS[ds]]
        Am = np.stack([z[f'{ds}|{bk}|A'][:, idx] for bk in config.BACKBONES])
        Sm = np.stack([z[f'{ds}|{bk}|S'] for bk in config.BACKBONES])
        n_subj = config.N_SUBJECTS[ds]
        r = dict(K=config.BUDGETS[ds], n_subj=n_subj,
                 main=curve_summary(Am, Sm, arms['calib'][2], n_subj),
                 **{w: curve_summary(*arms[w], n_subj) for w in WINDOWS})
        f, c = r['full'], r['calib']
        r['diff'] = {k: round(c[k] - f[k], 4) for k in ('src_pp', 'delta_pp', 'sd_between_pp',
                                                         'span_pp', 'mde_pp', 'ratio')}
        r['diff']['n_turned'] = c['n_turned'] - f['n_turned']
        d_src = np.array(c['per_arch_src_pp']) - np.array(f['per_arch_src_pp'])
        d_full = np.array(c['per_arch_full_pp']) - np.array(f['per_arch_full_pp'])
        r['paired'] = dict(d_src_pp=[round(v, 3) for v in d_src], d_full_pp=[round(v, 3) for v in d_full],
                           d_src_mean=float(d_src.mean()), d_full_mean=float(d_full.mean()),
                           p_src=float(stats.wilcoxon(d_src)[1]), p_full=float(stats.wilcoxon(d_full)[1]))
        res[ds] = r
        print(f'  {ds:<10} source {f["src_pp"]:.2f} → {c["src_pp"]:.2f}   Δ {f["delta_pp"]:+.2f} → '
              f'{c["delta_pp"]:+.2f}   positive backbones {f["n_turned"]} → {c["n_turned"]}')
    return res


def _pear(x, y):
    return float(np.corrcoef(x, y)[0, 1])


def _spear(x, y):
    return float(stats.spearmanr(x, y)[0])


def _perm_p(mx, my, f):
    obs = f(mx, my)
    return float(np.mean([f(mx, my[list(p)]) <= obs + 1e-12 for p in PERMS])), obs


def p4b(c_frozen):
    res = {}
    for w in WINDOWS:
        crv, src = {}, {}
        for ds in DS4:
            c = config.load_json(curves_file(ds, w))
            for bk, v in c['cells'].items():
                crv[f'{ds}|{bk}'] = {'N': c['K'], 'acc': list(np.mean(v['acc'], 0))}
                src[f'{ds}|{bk}'] = float(np.mean(v['src']))
        cs = CurveSet(crv)
        own = np.array([k.split('|')[0] for k in cs.keys])
        s = np.array([src[k] for k in cs.keys])
        res[w] = dict(n_units=len(cs.keys))
        for tag, c in (('frozen', c_frozen), ('refit', fit_shared_c(cs))):
            a, _, _ = cs.fit(c)
            h = a - s
            mx = np.array([s[own == d].mean() for d in DS4])
            my = np.array([h[own == d].mean() for d in DS4])
            p_r, r4 = _perm_p(mx, my, _pear)
            p_rho, rho4 = _perm_p(mx, my, _spear)
            sw = s - np.array([s[own == d].mean() for d in own])
            hw = h - np.array([h[own == d].mean() for d in own])
            res[w][tag] = dict(c=float(c), p4b_48=_pear(s, h), pooled_within=_pear(sw, hw),
                               within={d: _pear(s[own == d], h[own == d]) for d in DS4},
                               ds_src_pp=dict(zip(DS4, (100 * mx).tolist())),
                               ds_head_pp=dict(zip(DS4, (100 * my).tolist())), r4=r4, rho4=rho4,
                               perm_p_r=p_r, perm_p_rho=p_rho,
                               src_order=sorted(DS4, key=lambda d: -mx[DS4.index(d)]),
                               head_order=sorted(DS4, key=lambda d: -my[DS4.index(d)]),
                               p4b_pass=bool(_pear(s, h) <= -0.4))
        print(f'  P4b [{w}] frozen c: {res[w]["frozen"]["p4b_48"]:+.4f}   refit c: {res[w]["refit"]["p4b_48"]:+.4f}')
    return res


def by_ds(window):
    """{ds: [(N, mean curve, src), ...]} and the backbone names, in sorted name order."""
    out, names = {}, {}
    for ds in DS4:
        c = config.load_json(curves_file(ds, window))
        items = sorted(c['cells'].items())
        Ks = np.asarray(c['K'], float)
        out[ds] = [(Ks, np.asarray(v['acc'], float).mean(0), float(np.mean(v['src']))) for _, v in items]
        names[ds] = [k for k, _ in items]
    return out, names


def headroom_by_form():
    res = {}
    for w in WINDOWS:
        B, _ = by_ds(w)
        res[w] = {}
        for name in FORMS:
            pf = pooled_fit(B, DS4, name)
            hr = {ds: float(np.mean([h for h, o in zip(pf['hr'], pf['owner']) if o == ds])) for ds in DS4}
            th = pf['theta']
            res[w][name] = dict(theta=None if th is None else (float(th) if np.isscalar(th) else [float(x) for x in th]),
                                headroom_pp=hr, order=sorted(DS4, key=lambda d: -hr[d]))
            res[w][name]['order_ok'] = res[w][name]['order'] == ORDER
        res[w]['src_pp'] = {ds: float(100 * np.mean([s for _, _, s in B[ds]])) for ds in DS4}
    res['all_order_ok'] = all(res[w][n]['order_ok'] for w in WINDOWS for n in FORMS)
    res['range_pp'] = {ds: max(res[w][n]['headroom_pp'][ds] for w in WINDOWS for n in FORMS)
                       - min(res[w][n]['headroom_pp'][ds] for w in WINDOWS for n in FORMS) for ds in DS4}
    print(f'  headroom ordering 2a > Cho2017 > PhysioNet > OpenBMI in all 10 form × window cases: '
          f'{res["all_order_ok"]}')
    return res


def drop_deepconvnet():
    res = {}
    for w in WINDOWS:
        B, names = by_ds(w)
        unit = [u for ds in DS4 for u in names[ds]]
        res[w] = {}
        for name in FORMS:
            pf = pooled_fit(B, DS4, name)

            def means(keep):
                return {ds: float(np.mean([h for h, o, u in zip(pf['hr'], pf['owner'], unit)
                                           if o == ds and keep(u)])) for ds in DS4}
            h12, h11 = means(lambda u: True), means(lambda u: u != 'deepconvnet')
            res[w][name] = dict(headroom_pp_12=h12, headroom_pp_11=h11,
                                order_11=sorted(DS4, key=lambda d: -h11[d]))
            res[w][name]['order_ok_11'] = res[w][name]['order_11'] == ORDER
    PU = config.load_json('budget_law.json')['per_unit']
    main = {}
    for key, keep_deep in (('12', True), ('11', False)):
        main[key] = {ds: float(np.mean([(v['a'] - v['src_acc']) * 100 for v in PU.values()
                                        if v['ds'] == ds and (keep_deep or v['backbone'] != 'deepconvnet')]))
                     for ds in DS4}
    main['order_ok_11'] = sorted(DS4, key=lambda d: -main['11'][d]) == ORDER
    res['main_pow3'] = main
    res['all_order_ok_11'] = all(res[w][n]['order_ok_11'] for w in WINDOWS for n in FORMS) and main['order_ok_11']
    print(f'  ordering without DeepConvNet holds in all cases: {res["all_order_ok_11"]}')
    return res


def report():
    z = np.load(config.analysis_file('foldlevel_curves.npz'))
    c36 = round(config.load_json('preregistration_layerA.json')['meta']['baseline']['c_pooled_36'], 4)
    res = dict(table=table(z), p4b=p4b(c36), headroom=headroom_by_form(), drop_deepconvnet=drop_deepconvnet())
    print(f"[saved] {config.save_json('ea_window.json', res)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('step', choices=['features', 'curves', 'report'])
    ap.add_argument('--dataset', choices=DS4)
    ap.add_argument('--window', choices=WINDOWS)
    ap.add_argument('--backbones', nargs='+', default=config.BACKBONES, choices=config.BACKBONES)
    ap.add_argument('--folds', nargs='+', type=int, default=None)
    ap.add_argument('--validate', action='store_true',
                    help='compare the full window with the main features (source accuracy, features)')
    a = ap.parse_args()
    if a.step == 'report':
        report()
        return
    if not (a.dataset and a.window):
        ap.error('--dataset and --window are required')
    if a.step == 'features':
        features(a.dataset, a.window, a.backbones, set(a.folds) if a.folds else None, a.validate)
    else:
        curves(a.dataset, a.window)


if __name__ == '__main__':
    main()
