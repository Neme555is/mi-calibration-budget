# -*- coding: utf-8 -*-
"""Where the variation of the fine-tuning gain lies: cohorts, methods or subjects.

1  Variance components of Δ = head-fine-tuning accuracy − frozen-source accuracy (pp), a pure
   measurement without curve fitting. For every dataset a subject × backbone table is
   decomposed into backbone effects, subject effects and residuals; the spread of the dataset
   means (cohorts) is compared with the within-dataset spread over backbones, with a bootstrap
   over datasets and backbones (4000 draws). Done at the largest and at the smallest budget of
   the dense grid; the cell-level headroom is decomposed the same way under five forms.
2  Split-half reliability (2000 random splits, Spearman–Brown corrected) of the subject
   component and the backbone component, and the attenuation-corrected ratio of the two.
3  Agreement of subject difficulty across backbones: ICC(2,1) of the frozen-source accuracies
   and the mean pairwise Spearman correlation; the distribution of the OpenBMI per-subject
   source accuracy (averaged over backbones).

Inputs: foldlevel_curves.npz, budget_law.json. Output: heterogeneity.json

  python scripts/08_heterogeneity.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.stats import spearmanr

from calibration_budget import config
from calibration_budget.forms import FORMS, cells_with_src, fit_and_headroom

DS4 = config.DATASETS
SEED = 20260917


def load():
    z = np.load(config.analysis_file('foldlevel_curves.npz'), allow_pickle=True)
    ds_arch = {}
    for k in z.files:
        if k.endswith('|A'):
            ds, bk, _ = k.split('|')
            ds_arch.setdefault(ds, []).append(bk)
    return z, {ds: sorted(set(v)) for ds, v in ds_arch.items()}


def delta_matrix(z, ds, backbones, budget_idx=-1):
    """(subjects, backbones) table of Δ in pp."""
    return np.column_stack([100 * (z[f'{ds}|{b}|A'][:, budget_idx] - z[f'{ds}|{b}|S'])
                            for b in backbones])


def decompose(M):
    mu = M.mean()
    a = M.mean(0) - mu
    s = M.mean(1) - mu
    r = M - mu - a[:, None].T - s[:, None]
    return dict(mu=float(mu), sd_arch=float(a.std(ddof=1)), sd_subj=float(s.std(ddof=1)),
                sd_resid=float(r.std(ddof=1)))


def variance_components(z, archs, PU):
    report = {}
    for tag, bidx in (('max', -1), ('min', 0)):
        per_ds, Ms = {}, {}
        for ds in DS4:
            nb = z[f'{ds}|{archs[ds][0]}|A'].shape[1]
            M = delta_matrix(z, ds, archs[ds], min(bidx, nb - 1) if bidx >= 0 else bidx)
            per_ds[ds], Ms[ds] = decompose(M), M
        mus = np.array([per_ds[ds]['mu'] for ds in DS4])
        sd_cohort = float(mus.std(ddof=1))
        sd_arch = float(np.mean([per_ds[ds]['sd_arch'] for ds in DS4]))
        rng = np.random.default_rng(SEED)
        boots = []
        for _ in range(4000):
            mus_b, arch_b = [], []
            for j in rng.integers(0, len(DS4), len(DS4)):
                M = Ms[DS4[j]]
                Mb = M[:, rng.integers(0, M.shape[1], M.shape[1])]
                mus_b.append(Mb.mean())
                arch_b.append((Mb.mean(0) - Mb.mean()).std(ddof=1))
            boots.append((np.std(mus_b, ddof=1), np.mean(arch_b)))
        boots = np.array(boots)
        ratio = boots[:, 0] / boots[:, 1]
        report[tag] = dict(per_ds=per_ds, sd_cohort=sd_cohort, sd_arch=sd_arch,
                           ratio_median=float(np.median(ratio)),
                           ratio_ci=[float(np.percentile(ratio, 2.5)), float(np.percentile(ratio, 97.5))],
                           p_ratio_le1=float((ratio <= 1).mean()))
        print(f'  Δ at the {tag} budget: cohort SD {sd_cohort:.2f} pp vs backbone SD {sd_arch:.2f} pp '
              f'(ratio median {np.median(ratio):.2f})')
    rows = {}
    for name in FORMS:
        by_ds = {ds: cells_with_src(PU, ds) for ds in DS4}
        mus, sds = [], []
        for ds in DS4:
            hr, _, _ = fit_and_headroom(by_ds[ds], name)
            mus.append(hr.mean())
            sds.append(hr.std(ddof=1))
        sc, sa = float(np.std(mus, ddof=1)), float(np.mean(sds))
        rows[name] = dict(sd_cohort=sc, sd_arch=sa, ratio=sc / sa)
    report['headroom_cell'] = rows
    return report


def reliability(z, archs):
    def sb(r):
        return 2 * r / (1 + r)
    rng = np.random.default_rng(SEED)
    out = {}
    for ds in DS4:
        nb = z[f'{ds}|{archs[ds][0]}|A'].shape[1]
        for tag, bi in (('low', 0), ('mid', min(1, nb - 1)), ('full', -1)):
            M = delta_matrix(z, ds, archs[ds], bi)
            ns, na = M.shape
            d = decompose(M)
            rs, ra = [], []
            for _ in range(2000):
                p = rng.permutation(na)
                rs.append(np.corrcoef(M[:, p[:na // 2]].mean(1), M[:, p[na // 2:]].mean(1))[0, 1])
                q = rng.permutation(ns)
                ra.append(np.corrcoef(M[q[:ns // 2]].mean(0), M[q[ns // 2:]].mean(0))[0, 1])
            sb_s, sb_a = sb(float(np.mean(rs))), sb(float(np.mean(ra)))
            out[f'{ds}|{tag}'] = dict(
                n_subj=ns, n_arch=na, sd_subj=d['sd_subj'], sd_arch=d['sd_arch'],
                sb_sub=sb_s, sb_arc=sb_a, sig_sub=d['sd_subj'] * np.sqrt(sb_s),
                sig_arc=d['sd_arch'] * np.sqrt(sb_a), noise_share_sub=1 - sb_s,
                noise_share_arc=1 - sb_a,
                ratio=float(d['sd_subj'] * np.sqrt(sb_s) / (d['sd_arch'] * np.sqrt(sb_a))))
    n_gt = sum(v['ratio'] > 1 for v in out.values())
    print(f'  attenuation-corrected subject/backbone ratio > 1 in {n_gt}/{len(out)} cells')
    return out


def icc(z, archs):
    out = {}
    for ds in DS4:
        M = np.array([z[f'{ds}|{b}|S'] for b in archs[ds]])          # (backbones, subjects)
        E, N = M.shape
        g = M.mean()
        ms_b = E * ((M.mean(0) - g) ** 2).sum() / (N - 1)
        ms_w = ((M - M.mean(0)) ** 2).sum() / (N * (E - 1))
        rs = [spearmanr(M[i], M[j])[0] for i in range(E) for j in range(i + 1, E)]
        out[ds] = dict(n=N, n_arch=E, icc=float((ms_b - ms_w) / (ms_b + (E - 1) * ms_w)),
                       rho=float(np.mean(rs)), sd_between_pp=float(100 * M.mean(0).std(ddof=1)),
                       sd_within_pp=float(100 * M.std(0, ddof=1).mean()))
        print(f'  {ds:<10} ICC(2,1) of source accuracy = {out[ds]["icc"]:.3f}   mean Spearman = {out[ds]["rho"]:.3f}')
    per_subj = np.array([z[f'openbmi|{b}|S'] for b in archs['openbmi']]).mean(0)
    out['openbmi_per_subject'] = dict(
        n=int(len(per_subj)), mean=float(per_subj.mean()), median=float(np.median(per_subj)),
        sd=float(per_subj.std(ddof=1)), lo=float(per_subj.min()), hi=float(per_subj.max()),
        p10=float(np.percentile(per_subj, 10)), p25=float(np.percentile(per_subj, 25)),
        p75=float(np.percentile(per_subj, 75)), p90=float(np.percentile(per_subj, 90)),
        n_below_chance=int((per_subj < 0.5).sum()))
    return out


def main():
    z, archs = load()
    PU = config.load_json('budget_law.json')['per_unit']
    res = {'variance': variance_components(z, archs, PU), 'reliability': reliability(z, archs),
           'agreement': icc(z, archs)}
    print(f"[saved] {config.save_json('heterogeneity.json', res)}")


if __name__ == '__main__':
    main()
