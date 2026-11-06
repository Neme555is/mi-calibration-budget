# -*- coding: utf-8 -*-
"""Dataset-level summary of a set of fold-level budget curves."""
import numpy as np
from scipy import stats


def curve_summary(A, SRC, nev, n_subj):
    """Summary of fold-level accuracies A (backbones, folds, budgets).

    SRC (backbones, folds) are the frozen-source accuracies and nev (folds,) the evaluation
    block sizes. Returns the source accuracy, the full-budget gain Δ, the number of backbones
    with Δ > 0, the between-subject SD and binomial share of the between-fold variance, the
    span of the mean curve, the minimum detectable difference at 80 % power and span/MDE.
    """
    E, nf, nK = A.shape
    tot, bino = [], []
    for e in range(E):
        for j in range(nK):
            a = A[e, :, j]
            tot.append(float(a.var(ddof=1)))
            bino.append(float(np.mean(a * (1.0 - a) / nev)))
    tot, bino = np.array(tot), np.array(bino)
    bet = np.maximum(tot - bino, 0.0)
    sd_between_pp = float(100 * np.sqrt(bet.mean()))
    curve = A.mean((0, 1))
    per_full, per_src = A[:, :, -1].mean(1), SRC.mean(1)
    ts = float(stats.t.ppf(0.975, nf - 1) + stats.t.ppf(0.80, nf - 1))
    mde = float(ts * sd_between_pp / 100.0 / np.sqrt(nf) * 100)
    span = float(100 * (curve[-1] - curve[0]))
    return dict(n_arch=int(E), n_fold=int(nf), src_pp=float(100 * SRC.mean()),
                delta_pp=float(100 * (curve[-1] - SRC.mean())), n_turned=int((per_full > per_src).sum()),
                sd_between_pp=sd_between_pp, binom_share=float(bino.sum() / tot.sum()),
                span_pp=span, mde_pp=mde, ratio=float(span / mde), curve_pp=[float(100 * c) for c in curve],
                per_arch_full_pp=[float(100 * v) for v in per_full],
                per_arch_src_pp=[float(100 * v) for v in per_src])
