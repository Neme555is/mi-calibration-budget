# -*- coding: utf-8 -*-
"""How much does the dataset ordering of the headroom depend on which backbones are averaged?

The per-unit headroom comes from the pooled fit of all 48 units (five forms × two EA windows,
as in scripts/13_ea_window.py); for every subset of k backbones (k = 4, 6, 8, 10, 12; all
C(12, k) subsets) the dataset means are recomputed without refitting, and the frequency of each
ordering is counted: 2a first, Cho2017 second, both, and PhysioNet above OpenBMI.

Inputs: budget_law.json (full window), ea_window_curves_<ds>_calib.json (calibration window).
Output: ordering_subsets.json

  python scripts/14_ordering_subsets.py
"""
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from calibration_budget import config
from calibration_budget.forms import FORMS, pooled_fit

DS4 = config.DATASETS


def unit_headroom(window, name, PU):
    """{dataset: per-backbone headroom (pp)}, backbones in sorted name order."""
    if window == 'full':
        B = {ds: [(np.asarray(PU[k]['N'], float), np.asarray(PU[k]['acc'], float), float(PU[k]['src_acc']))
                  for k in sorted(k for k in PU if k.startswith(ds + '|'))] for ds in DS4}
    else:
        B = {}
        for ds in DS4:
            c = config.load_json(f'ea_window_curves_{ds}_{window}.json')
            Ks = np.asarray(c['K'], float)
            B[ds] = [(Ks, np.asarray(v['acc'], float).mean(0), float(np.mean(v['src'])))
                     for _, v in sorted(c['cells'].items())]
    pf = pooled_fit(B, DS4, name)
    hr, own = np.asarray(pf['hr']), np.asarray(pf['owner'])
    return {ds: hr[own == ds] for ds in DS4}


def main():
    PU = config.load_json('budget_law.json')['per_unit']
    res = {}
    for window in ('full', 'calib'):
        for name in FORMS:
            H = unit_headroom(window, name, PU)
            n = len(H['2a'])
            out = {}
            for k in (4, 6, 8, 10, 12):
                tally, first, second, ph_gt_ob, tiers = {}, 0, 0, 0, 0
                subs = list(itertools.combinations(range(n), k))
                for s in subs:
                    m = {ds: float(H[ds][list(s)].mean()) for ds in DS4}
                    o = tuple(sorted(DS4, key=lambda d: -m[d]))
                    tally[o] = tally.get(o, 0) + 1
                    first += o[0] == '2a'
                    second += o[1] == 'cho2017'
                    ph_gt_ob += m['physionet'] > m['openbmi']
                    tiers += (o[0] == '2a' and o[1] == 'cho2017')
                N = len(subs)
                out[str(k)] = dict(n_subsets=N, p_2a_first=first / N, p_cho_second=second / N,
                                   p_tiers=tiers / N, p_phys_gt_openbmi=ph_gt_ob / N,
                                   top_orders={'>'.join(o): c / N for o, c in
                                               sorted(tally.items(), key=lambda t: -t[1])[:4]})
            res[f'{window}|{name}'] = out
            o6 = out['6']
            print(f'  {window:5s} {name:6s} k=6: 2a first {o6["p_2a_first"]:.3f}  Cho2017 second '
                  f'{o6["p_cho_second"]:.3f}  PhysioNet > OpenBMI {o6["p_phys_gt_openbmi"]:.3f}')
    print(f"[saved] {config.save_json('ordering_subsets.json', res)}")


if __name__ == '__main__':
    main()
