# -*- coding: utf-8 -*-
"""Saturating functional forms compared with the power law.

Every form is acc = a_e + β_e · f(N; θ) with a_e, β_e free per unit and the shape parameter θ
shared across units (the power law's b in the paper is −β):

    pow3   f = N^(−c)                  one shared parameter
    exp3   f = exp(−N/τ)               one shared parameter
    wei3   f = exp(−(N/τ)^d)           two shared parameters (Weibull)
    ilog2  f = 1/ln N                  none
    pow2   f = N^(−1/2)                none (power law with c fixed at 1/2)
    log2   f = ln N                    none (no asymptote)
    freec  N^(−c_e) with c_e free per unit (saturated with three budgets per unit)

For a given θ the per-unit coefficients are linear least squares; θ is found on a grid.
The headroom of a unit is its asymptote minus its frozen-source accuracy, a_e − src_e.
"""
import numpy as np


def f_pow3(N, th):
    return N ** (-th)


def f_exp3(N, th):
    return np.exp(-N / th)


def f_wei3(N, th):
    tau, d = th
    return np.exp(-((N / tau) ** d))


def f_ilog2(N, th):
    return 1.0 / np.log(N)


def f_pow2(N, th):
    return N ** (-0.5)


def f_log2(N, th):
    return np.log(N)


MODELS = {
    'pow3': dict(f=f_pow3, dim=1, label='power law  a + b·N^(−c)'),
    'exp3': dict(f=f_exp3, dim=1, label='exponential  a + b·exp(−N/τ)'),
    'wei3': dict(f=f_wei3, dim=2, label='Weibull  a + b·exp(−(N/τ)^d)'),
    'ilog2': dict(f=f_ilog2, dim=0, label='a + b/ln N'),
    'pow2': dict(f=f_pow2, dim=0, label='a + b·N^(−1/2)  (c fixed at 1/2)'),
    'log2': dict(f=f_log2, dim=0, label='a + b·ln N'),
}
# Forms with an asymptote (log2 has none, so its headroom is undefined).
FORMS = ['pow3', 'exp3', 'wei3', 'pow2', 'ilog2']

GRID_C = np.exp(np.linspace(np.log(0.005), np.log(4.0), 1600))
GRID_TAU = np.exp(np.linspace(np.log(0.3), np.log(800.0), 1600))
GRID_D = np.linspace(0.25, 3.0, 40)


def grid_for(name):
    if name == 'wei3':
        return [(t, d) for t in GRID_TAU[::16] for d in GRID_D]
    if name == 'exp3':
        return GRID_TAU
    if name == 'pow3':
        return GRID_C
    return [None]


def fit_shared(cells, name):
    """Best shared shape parameter. cells = [(N, y), ...]; returns (sse, theta, coefs)."""
    spec = MODELS[name]
    best = (np.inf, None, None)
    for th in grid_for(name):
        sse = 0.0
        coefs = []
        for N, y in cells:
            F = np.column_stack([np.ones_like(N), spec['f'](N, th)])
            coef, *_ = np.linalg.lstsq(F, y, rcond=None)
            r = y - F @ coef
            sse += float(r @ r)
            coefs.append(coef)
        if sse < best[0]:
            best = (sse, th, coefs)
    return best


def fit_free_c(cells):
    """Power law with a free exponent per unit."""
    sse = 0.0
    coefs = []
    for N, y in cells:
        best = (np.inf, None)
        for c in GRID_C:
            F = np.column_stack([np.ones_like(N), N ** (-c)])
            coef, *_ = np.linalg.lstsq(F, y, rcond=None)
            r = y - F @ coef
            s = float(r @ r)
            if s < best[0]:
                best = (s, (c, coef))
        sse += best[0]
        coefs.append(best[1])
    return sse, None, coefs


def aicc(sse, n, k):
    if n <= k + 1:
        return float('inf')
    return n * np.log(sse / n) + 2 * k + 2 * k * (k + 1) / (n - k - 1)


def theta_json(th):
    if th is None:
        return None
    return float(th) if np.isscalar(th) else [float(x) for x in th]


# ── Headroom under each form ──────────────────────────────────────────────────────────
def cells_with_src(per_unit, dataset):
    """[(N, acc, src), ...] of one dataset, units in sorted key order."""
    out = []
    for k in sorted(per_unit):
        v = per_unit[k]
        if v['ds'] != dataset:
            continue
        out.append((np.asarray(v['N'], float), np.asarray(v['acc'], float), float(v['src_acc'])))
    return out


def fit_and_headroom(cells, name, theta=None):
    """Per-unit headroom (pp) and β of one dataset fitted on its own."""
    spec = MODELS[name]
    if theta is None:
        _, theta, coefs = fit_shared([(N, y) for N, y, _ in cells], name)
    else:
        coefs = []
        for N, y, _ in cells:
            F = np.column_stack([np.ones_like(N), spec['f'](N, theta)])
            coef, *_ = np.linalg.lstsq(F, y, rcond=None)
            coefs.append(coef)
    hr = np.array([(c[0] - s) * 100 for c, (_, _, s) in zip(coefs, cells)])
    b = np.array([c[1] for c in coefs])
    return hr, b, theta


def pooled_fit(by_ds, datasets, name, theta=None):
    """One shared shape parameter over all units of ``datasets`` (the paper's pooled fit).

    Returns per-unit a, b (paper sign, b = −β), source accuracy, headroom (pp), the dataset of
    each unit and θ.
    """
    allc, owner = [], []
    for ds in datasets:
        for c in by_ds[ds]:
            allc.append(c)
            owner.append(ds)
    spec = MODELS[name]
    if theta is None:
        _, theta, coefs = fit_shared([(N, y) for N, y, _ in allc], name)
    else:
        coefs = []
        for N, y, _ in allc:
            F = np.column_stack([np.ones_like(N), spec['f'](N, theta)])
            coef, *_ = np.linalg.lstsq(F, y, rcond=None)
            coefs.append(coef)
    a = np.array([c[0] for c in coefs])
    b = np.array([-c[1] for c in coefs])
    src = np.array([s for _, _, s in allc])
    return dict(a=a, b=b, src=src, hr=(a - src) * 100, owner=owner, theta=theta)


def headroom_at(curve, Ks, src, name, theta):
    """Headroom (pp) of a single mean curve for a fixed shape parameter."""
    spec = MODELS[name]
    N = np.asarray(Ks, float)
    F = np.column_stack([np.ones_like(N), spec['f'](N, theta)])
    coef, *_ = np.linalg.lstsq(F, np.asarray(curve, float), rcond=None)
    return float((coef[0] - src) * 100)
