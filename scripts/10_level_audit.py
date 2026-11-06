# -*- coding: utf-8 -*-
"""Level audit: the 48 units are nested in 4 datasets.

1  P4b (corr of source accuracy with headroom over 48 units) is a between-dataset quantity:
   each dataset contributes one (source, headroom) point, so the effective n is 4. Reported:
   the 48-unit correlation, the within-dataset correlations, the pooled within-dataset
   correlation, the 4-point correlation with its exact permutation p (24 pairings; 1/24 is
   the smallest attainable one-sided p), and a dataset-level cluster bootstrap (5000 draws).
   Computed at the frozen three-dataset exponent and at the 48-unit exponent, and for the
   five functional forms.
2  Intervals for c: unit-level bootstrap (as in the paper), two-stage bootstrap (datasets,
   then units), dataset-block bootstrap, leave-one-dataset-out fits and a random-effects
   interval over the four per-dataset exponents.

Inputs: budget_law.json, functional_forms.json, preregistration_layerA.json.
Output: level_audit.json

  python scripts/10_level_audit.py
"""
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy import stats

from calibration_budget import config
from calibration_budget.curves import CurveSet, fit_shared_c
from calibration_budget.forms import FORMS

DS4 = config.DATASETS
N_BOOT = 2000
PERMS = list(itertools.permutations(range(4)))


def _pear(x, y):
    return float(np.corrcoef(x, y)[0, 1])


def _spear(x, y):
    return float(stats.spearmanr(x, y)[0])


def _perm_p(mx, my, stat):
    """Exact one-sided (negative) permutation p over the 24 pairings of four points."""
    obs = stat(mx, my)
    return float(np.mean([stat(mx, my[list(p)]) <= obs + 1e-12 for p in PERMS])), obs


def main():
    D = config.load_json('budget_law.json')
    HR = config.load_json('functional_forms.json')['headroom']
    c36 = round(config.load_json('preregistration_layerA.json')['meta']['baseline']['c_pooled_36'], 4)
    PU = D['per_unit']
    keys = list(PU)
    ds_of = np.array([k.split('|')[0] for k in keys])
    src = np.array([PU[k]['src_acc'] for k in keys])
    cs = CurveSet({k: {'N': PU[k]['N'], 'acc': PU[k]['acc']} for k in keys})
    idx_of = {d: np.array([i for i, k in enumerate(cs.keys) if k.split('|')[0] == d]) for d in DS4}

    def level_audit(x, y):
        mx = np.array([x[ds_of == d].mean() for d in DS4])
        my = np.array([y[ds_of == d].mean() for d in DS4])
        xw = x - np.array([x[ds_of == d].mean() for d in ds_of])
        yw = y - np.array([y[ds_of == d].mean() for d in ds_of])
        p_r, r4 = _perm_p(mx, my, _pear)
        p_rho, rho4 = _perm_p(mx, my, _spear)
        t4 = r4 * np.sqrt(2) / np.sqrt(1 - r4 ** 2)
        r48 = _pear(x, y)
        t48 = r48 * np.sqrt(len(x) - 2) / np.sqrt(1 - r48 ** 2)
        return dict(r_48=r48, p_if_n48=float(2 * stats.t.sf(abs(t48), len(x) - 2)),
                    within={d: _pear(x[ds_of == d], y[ds_of == d]) for d in DS4},
                    r_pooled_within=_pear(xw, yw), ds_src=(mx * 100).tolist(), ds_head=(my * 100).tolist(),
                    r_4=r4, p_4_parametric=float(2 * stats.t.sf(abs(t4), 2)), rho_4=rho4,
                    perm_p_r=p_r, perm_p_rho=p_rho, monotone=bool(my[0] > my[1] > my[2] > my[3]))

    a36, _, _ = cs.fit(c36)
    A = level_audit(src, a36 - src)
    a48, _, _ = cs.fit(D['pooled']['c'])
    B = level_audit(src, a48 - src)
    rng = np.random.default_rng(20260920)
    bb = []
    for _ in range(5000):
        pick = rng.integers(0, 4, 4)
        xs = np.concatenate([src[ds_of == DS4[i]] for i in pick])
        ys = np.concatenate([(a36 - src)[ds_of == DS4[i]] for i in pick])
        if np.std(xs) > 1e-12 and np.std(ys) > 1e-12:
            bb.append(_pear(xs, ys))
    bb = np.array(bb)
    r_ci = [float(np.percentile(bb, 2.5)), float(np.percentile(bb, 97.5))]
    print(f'P4b at c = {c36}: r(48 units) = {A["r_48"]:+.4f}; r(4 datasets) = {A["r_4"]:+.4f}, '
          f'permutation p = {A["perm_p_r"]:.4f}; pooled within-dataset r = {A["r_pooled_within"]:+.4f}')

    mx = np.array([src[ds_of == d].mean() for d in DS4]) * 100
    by_form = {}
    for f in FORMS:
        my = np.array([HR['by_form'][f][d] for d in DS4])
        p_r, r4 = _perm_p(mx, my, _pear)
        by_form[f] = dict(r_48=HR['structure'][f]['p4b'], r_4=r4, rho_4=_spear(mx, my), perm_p_r=p_r,
                          monotone=bool(my[0] > my[1] > my[2] > my[3]), head=my.tolist())

    c_pool = fit_shared_c(cs)
    c_ds = {d: fit_shared_c(cs, idx=idx_of[d]) for d in DS4}
    vals = np.array([c_ds[d] for d in DS4])
    n = len(cs.keys)
    rng = np.random.default_rng(42)
    b_unit = np.array([fit_shared_c(cs, idx=rng.integers(0, n, n)) for _ in range(N_BOOT)])
    ci_unit = [float(np.percentile(b_unit, 2.5)), float(np.percentile(b_unit, 97.5))]
    rng = np.random.default_rng(20260920)
    b_two = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, 4, 4)
        idx = np.concatenate([rng.choice(idx_of[DS4[i]], len(idx_of[DS4[i]]), replace=True) for i in pick])
        b_two.append(fit_shared_c(cs, idx=idx))
    ci_two = [float(np.percentile(b_two, 2.5)), float(np.percentile(b_two, 97.5))]
    rng = np.random.default_rng(20260920)
    b_blk = [fit_shared_c(cs, idx=np.concatenate([idx_of[DS4[i]] for i in rng.integers(0, 4, 4)]))
             for _ in range(N_BOOT)]
    ci_blk = [float(np.percentile(b_blk, 2.5)), float(np.percentile(b_blk, 97.5))]
    lodo = {d: fit_shared_c(cs, idx=np.concatenate([idx_of[e] for e in DS4 if e != d])) for d in DS4}
    half = float(stats.t.ppf(0.975, 3) * vals.std(ddof=1) / np.sqrt(4))
    print(f'c = {c_pool:.4f}; unit bootstrap [{ci_unit[0]:.3f}, {ci_unit[1]:.3f}], '
          f'two-stage [{ci_two[0]:.3f}, {ci_two[1]:.3f}], dataset blocks [{ci_blk[0]:.3f}, {ci_blk[1]:.3f}]')

    out = dict(p4b=dict(frozen=A, current=B, c_frozen_36=c36, by_form=by_form, cluster_boot_ci=r_ci,
                        frac_boot_above_thr=float((bb > -0.4).mean()), min_attainable_perm_p=1 / 24),
               c=dict(pooled=c_pool, per_dataset=c_ds, sd_between=float(vals.std(ddof=1)),
                      ratio=float(vals.max() / vals.min()), ci_unit=ci_unit, ci_two_stage=ci_two,
                      ci_dataset_block=ci_blk,
                      width_ratio=float((ci_two[1] - ci_two[0]) / (ci_unit[1] - ci_unit[0])),
                      lodo=lodo, re_mean=float(vals.mean()),
                      re_ci=[float(vals.mean() - half), float(vals.mean() + half)]))
    print(f"[saved] {config.save_json('level_audit.json', out)}")


if __name__ == '__main__':
    main()
