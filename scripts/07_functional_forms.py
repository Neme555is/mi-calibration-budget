# -*- coding: utf-8 -*-
"""Is the power law selected by the data, and does the headroom depend on the form?

Part 1 (AICc). The power law with a shared exponent is compared with other saturating forms
(``calibration_budget.forms``) on the same points, each with free per-unit a_e and b_e and
shared shape parameters: pooled over the four datasets at three budgets, per dataset, and on
the five-budget curves of scripts/06_exponent_identifiability.py (pooled and per dataset).
AICc = n·ln(SSE/n) + 2k + 2k(k+1)/(n−k−1). Also reported: the largest difference between the
power-law fit and each alternative inside the measured budget range and at 2× and 10× the
budget cap (per unit, median and maximum).

Part 2 (headroom). The headroom a_e − src_e is re-estimated under the five forms that have an
asymptote: dataset means (pooled fit of all 48 units, the paper's convention, and per-dataset
fits), the correlations that were preregistered (P3: headroom–b within a dataset; P4a:
source–headroom within a dataset; P4b: source–headroom over all units), the range of the
dataset means when the shape parameter sweeps a plausible range, and a unit-level cluster
bootstrap (2000 draws) with the shape parameter fixed.

Inputs: budget_law.json, exponent_identifiability.json. Output: functional_forms.json

  python scripts/07_functional_forms.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from calibration_budget import config
from calibration_budget.forms import (FORMS, MODELS, aicc, cells_with_src, f_pow3, fit_and_headroom,
                                      fit_free_c, fit_shared, pooled_fit, theta_json)

DS4 = config.DATASETS


def collect_cells(per_unit, datasets=None):
    cells = []
    for k in sorted(per_unit):
        v = per_unit[k]
        if datasets is None or v['ds'] in datasets:
            cells.append((np.asarray(v['N'], float), np.asarray(v['acc'], float)))
    return cells


def run_block(title, cells):
    n = sum(len(y) for _, y in cells)
    E = len(cells)
    rows = []
    for name in MODELS:
        sse, th, _ = fit_shared(cells, name)
        k = 2 * E + MODELS[name]['dim']
        rows.append((name, sse, k, aicc(sse, n, k), th))
    sse_f, _, _ = fit_free_c(cells)
    rows.append(('freec', sse_f, 3 * E, aicc(sse_f, n, 3 * E), None))
    base = min(r[3] for r in rows if np.isfinite(r[3]))
    out = []
    for name, sse, k, a, th in rows:
        fin = bool(np.isfinite(a))
        out.append(dict(name=name, k=k, sse=sse, aicc=a if fin else None,
                        daicc=(a - base) if fin else None, theta=theta_json(th) if fin else None))
    best = min((r for r in out if r['daicc'] is not None), key=lambda r: r['daicc'])
    print(f'  {title:<44} n={n:<4} best: {best["name"]}', flush=True)
    return out


def prediction_gap(cells):
    fits = {name: fit_shared(cells, name)[1:] for name in MODELS}
    th0, coefs0 = fits['pow3']
    in_range, out_range = [], []
    for i, (N, _) in enumerate(cells):
        grid_in = np.linspace(N.min(), N.max(), 40)
        grid_out = np.array([N.max() * 2.0, N.max() * 10.0])
        for name in ('exp3', 'wei3', 'ilog2', 'pow2', 'log2'):
            th, coefs = fits[name]
            f = MODELS[name]['f']
            ref_in = coefs0[i][0] + coefs0[i][1] * f_pow3(grid_in, th0)
            ref_out = coefs0[i][0] + coefs0[i][1] * f_pow3(grid_out, th0)
            in_range.append(float(np.max(np.abs(ref_in - (coefs[i][0] + coefs[i][1] * f(grid_in, th))))))
            out_range.append(float(np.max(np.abs(ref_out - (coefs[i][0] + coefs[i][1] * f(grid_out, th))))))
    return dict(in_range_max_pp=100 * float(np.max(in_range)),
                in_range_median_pp=100 * float(np.median(in_range)),
                out_range_max_pp=100 * float(np.max(out_range)),
                out_range_median_pp=100 * float(np.median(out_range)))


def aicc_part(PU):
    print('Part 1: AICc across forms')
    rep = {'pooled_3pt': run_block('four datasets pooled, 3 budgets', collect_cells(PU))}
    for ds in DS4:
        rep[f'{ds}_3pt'] = run_block(f'{config.DATASET_LABEL[ds]}, 3 budgets', collect_cells(PU, {ds}))
    idt = config.analysis_file('exponent_identifiability.json')
    if os.path.exists(idt):
        ID = config.load_json('exponent_identifiability.json')
        dense, keys = [], []
        for ds in ('2a', 'cho2017', 'physionet'):
            for key, curve in ID[ds]['curves'].items():
                arr = np.asarray(curve, float)
                dense.append((arr[:, 0], arr[:, 1]))
                keys.append(key)
        rep['pooled_5pt'] = run_block('three datasets pooled, 5 budgets', dense)
        by_ds = {}
        for key, cell in zip(keys, dense):
            by_ds.setdefault(key.split('|')[0], []).append(cell)
        for ds, cs in sorted(by_ds.items()):
            rep[f'{ds}_5pt'] = run_block(f'{ds}, 5 budgets', cs)
    rep['prediction_gap'] = prediction_gap(collect_cells(PU))
    g = rep['prediction_gap']
    print(f'  inside the budget range: median {g["in_range_median_pp"]:.2f} pp, max {g["in_range_max_pp"]:.2f} pp; '
          f'at 2x and 10x the cap: median {g["out_range_median_pp"]:.2f} pp, max {g["out_range_max_pp"]:.2f} pp')
    return rep


def headroom_part(PU):
    print('Part 2: headroom under five forms')
    by_ds = {ds: cells_with_src(PU, ds) for ds in DS4}
    report = {}
    for key, per_ds in (('by_form', False), ('by_form_perds', True)):
        rows = {}
        for name in FORMS:
            if per_ds:
                rows[name] = {ds: float(fit_and_headroom(by_ds[ds], name)[0].mean()) for ds in DS4}
            else:
                pf = pooled_fit(by_ds, DS4, name)
                rows[name] = {ds: float(pf['hr'][[i for i, o in enumerate(pf['owner']) if o == ds]].mean())
                              for ds in DS4}
        report[key] = rows
    for name in FORMS:
        print(f'  {name:<6} ' + '  '.join(f'{ds} {report["by_form"][name][ds]:6.2f}' for ds in DS4))

    struct = {}
    for name in FORMS:
        pf = pooled_fit(by_ds, DS4, name)
        p3, p4a = {}, {}
        for ds in DS4:
            m = np.array([o == ds for o in pf['owner']])
            p3[ds] = float(np.corrcoef(pf['hr'][m], pf['b'][m])[0, 1])
            p4a[ds] = float(np.corrcoef(pf['src'][m], pf['hr'][m])[0, 1])
        struct[name] = dict(p3=p3, p4a=p4a, p4b=float(np.corrcoef(pf['src'], pf['hr'])[0, 1]))
    report['structure'] = struct

    sweep = {}
    for name, grid in (('pow3', np.linspace(0.10, 0.50, 81)),
                       ('exp3', np.exp(np.linspace(np.log(2.0), np.log(120.0), 81)))):
        acc = {ds: [] for ds in DS4}
        for th in grid:
            pf = pooled_fit(by_ds, DS4, name, theta=th)
            for ds in DS4:
                m = np.array([o == ds for o in pf['owner']])
                acc[ds].append(float(pf['hr'][m].mean()))
        sweep[name] = {ds: [float(min(acc[ds])), float(max(acc[ds]))] for ds in DS4}
    report['shape_sweep'] = sweep

    rng = np.random.default_rng(20260917)
    boot = {}
    for name in FORMS:
        pf0 = pooled_fit(by_ds, DS4, name)
        point = {ds: float(pf0['hr'][[i for i, o in enumerate(pf0['owner']) if o == ds]].mean()) for ds in DS4}
        draws = {ds: [] for ds in DS4}
        for _ in range(2000):
            sub = {ds: [by_ds[ds][j] for j in rng.integers(0, len(by_ds[ds]), len(by_ds[ds]))] for ds in DS4}
            pf = pooled_fit(sub, DS4, name, theta=pf0['theta'])
            for ds in DS4:
                m = np.array([o == ds for o in pf['owner']])
                draws[ds].append(float(pf['hr'][m].mean()))
        boot[name] = dict(point=point, ci95={ds: [float(np.percentile(draws[ds], 2.5)),
                                                  float(np.percentile(draws[ds], 97.5))] for ds in DS4})
    report['bootstrap'] = boot
    report['cho_vs_phys'] = {n: [report['by_form'][n]['cho2017'], report['by_form'][n]['physionet']]
                             for n in FORMS}
    return report


def main():
    PU = config.load_json('budget_law.json')['per_unit']
    res = {'aicc': aicc_part(PU), 'headroom': headroom_part(PU)}
    print(f"[saved] {config.save_json('functional_forms.json', res)}")


if __name__ == '__main__':
    main()
