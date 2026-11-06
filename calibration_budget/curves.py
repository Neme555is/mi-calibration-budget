# -*- coding: utf-8 -*-
"""Budget–accuracy curves and the shared-exponent power law.

Model (one curve per backbone × dataset unit e):

    acc_e(N) = a_e − b_e · N^(−c),

with a_e, b_e free per unit and the exponent c shared. For a given c the best a_e, b_e are
closed-form least squares, so c is found by a grid search (step 0.002 on [0.02, 1.5)
followed by a 0.0001 step within ±0.01 of the coarse optimum). Writing the budget as N or
as N/D_e gives the same fit because b_e absorbs D_e^c; D matters only when b is shared.

R² is 1 − SSE/SST with SST taken around the grand mean of all points in the fit (between-unit
variance included).
"""
import numpy as np


class CurveSet:
    """{key: {'N': [...], 'acc': [...]}} packed into (E, n) arrays (equal n per unit)."""

    def __init__(self, sub):
        self.keys = list(sub.keys())
        self.N = np.array([sub[k]['N'] for k in self.keys], float)
        self.Y = np.array([sub[k]['acc'] for k in self.keys], float)
        self.n = self.N.shape[1]
        self.Ybar = self.Y.mean(1, keepdims=True)
        self.Syy = ((self.Y - self.Ybar) ** 2).sum(1)

    def sst(self, idx=None):
        Y = self.Y if idx is None else self.Y[idx]
        return float(((Y - Y.mean()) ** 2).sum())

    def _ab_sse(self, X, Y=None, Syy=None):
        Y = self.Y if Y is None else Y
        Syy = self.Syy if Syy is None else Syy
        Xc = X - X.mean(1, keepdims=True)
        Yc = Y - Y.mean(1, keepdims=True)
        Sxx = (Xc ** 2).sum(1)
        Sxy = (Xc * Yc).sum(1)
        b = -Sxy / Sxx
        a = Y.mean(1) + b * X.mean(1)
        sse = np.maximum(Syy - Sxy ** 2 / Sxx, 0.0)
        return a, b, sse

    def fit(self, c, idx=None, Y=None, Syy=None):
        """(a, b, SSE) per unit for exponent c; ``idx`` selects units (rows)."""
        N = self.N if idx is None else self.N[idx]
        if Y is None:
            Y = self.Y if idx is None else self.Y[idx]
        if Syy is None:
            Syy = self.Syy if idx is None else self.Syy[idx]
        return self._ab_sse(N ** (-c), Y=Y, Syy=Syy)

    def sse(self, c, idx=None, Y=None, Syy=None):
        return float(self.fit(c, idx, Y, Syy)[2].sum())


def fit_shared_c(cs, lo=0.02, hi=1.50, idx=None):
    grid = np.arange(lo, hi, 0.002)
    vals = [cs.sse(c, idx) for c in grid]
    c0 = float(grid[int(np.argmin(vals))])
    fine = np.arange(max(0.01, c0 - 0.01), c0 + 0.01, 0.0001)
    vf = [cs.sse(c, idx) for c in fine]
    return float(fine[int(np.argmin(vf))])


def r2(cs, c, idx=None):
    return 1.0 - cs.sse(c, idx) / cs.sst(idx)


# ── Least-squares variant on {key: [(N, acc), ...]} (used by the control analyses) ─────────
def sse_points(sub, c):
    tot = 0.0
    for pts in sub.values():
        N = np.array([p[0] for p in pts], float)
        y = np.array([p[1] for p in pts], float)
        x = N ** (-c)
        A = np.vstack([np.ones_like(x), -x]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        tot += float(np.sum((y - A @ coef) ** 2))
    return tot


def fit_c_points(sub, lo=0.02, hi=1.50):
    g = np.arange(lo, hi, 0.002)
    c0 = float(g[int(np.argmin([sse_points(sub, c) for c in g]))])
    f = np.arange(max(0.01, c0 - 0.01), c0 + 0.01, 0.0001)
    return float(f[int(np.argmin([sse_points(sub, c) for c in f]))])


def r2_points(sub, c):
    y = np.concatenate([np.array([p[1] for p in v]) for v in sub.values()])
    sst = float(np.sum((y - y.mean()) ** 2))
    return float('nan') if sst <= 0 else 1.0 - sse_points(sub, c) / sst


def params_points(sub, c):
    out = {}
    for k, pts in sub.items():
        N = np.array([p[0] for p in pts], float)
        y = np.array([p[1] for p in pts], float)
        x = N ** (-c)
        A = np.vstack([np.ones_like(x), -x]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        r = (y - A @ coef) * 100
        out[k] = dict(a=float(coef[0]), b=float(coef[1]), rmse_pp=float(np.sqrt(np.mean(r ** 2))))
    return out


# ── Vectorized fits for bootstrap replicates and ragged budget grids ─────────────────────
def pack(Ns_list, Yb, n_max=None):
    """Pad units with different grid lengths to (B, E, n_max) with a weight mask.

    Padding repeats the last point with weight 0, so weighted means equal the unpadded ones.
    ``Yb`` is (B, E, n) with unit e occupying the first len(Ns_list[e]) columns.
    """
    n_max = n_max or max(len(x) for x in Ns_list)
    B, E = Yb.shape[0], len(Ns_list)
    Np = np.zeros((E, n_max))
    W = np.zeros((E, n_max))
    Yp = np.zeros((B, E, n_max))
    for e, Ns in enumerate(Ns_list):
        n = len(Ns)
        Np[e, :n] = Ns
        Np[e, n:] = Ns[-1]
        W[e, :n] = 1.0
        Yp[:, e, :n] = Yb[:, e, :n]
        Yp[:, e, n:] = Yb[:, e, n - 1:n]
    return Yp, Np, W


def fit_c_boot(Yp, Np, W, grid):
    """Shared exponent of each replicate, (B,), by grid search over ``grid``."""
    B = Yp.shape[0]
    ne = W.sum(1)
    ybar = (W[None] * Yp).sum(2) / ne[None]
    Yc = W[None] * (Yp - ybar[:, :, None])
    Syy = (Yc ** 2).sum(2)
    best = np.full(B, np.inf)
    arg = np.zeros(B)
    for c in grid:
        X = Np ** (-c)
        xbar = (W * X).sum(1) / ne
        Xc = W * (X - xbar[:, None])
        Sxx = (Xc ** 2).sum(1)
        Sxy = np.einsum('en,ben->be', Xc, Yc)
        sse = Syy - Sxy ** 2 / Sxx[None]
        tot = np.maximum(sse, 0.0).sum(1)
        upd = tot < best
        best = np.where(upd, tot, best)
        arg = np.where(upd, c, arg)
    return arg


def fit_ab(Y, Ns, c):
    """Least-squares a, b of each row of Y (m, n) for exponent c (scalar or (m,))."""
    Ns = np.asarray(Ns, float)
    c = np.asarray(c, float)
    x = (Ns ** (-c))[None, :] if c.ndim == 0 else Ns[None, :] ** (-c[:, None])
    xc = x - x.mean(1, keepdims=True)
    yc = Y - Y.mean(1, keepdims=True)
    b = -(yc * xc).sum(1) / (xc ** 2).sum(1)
    a = Y.mean(1) + b * x.mean(1)
    return a, b


def fit_c_grid_shared(Ns_list, Y):
    """Shared exponent of mean curves Y (E, n) on possibly different budget grids."""
    g = np.arange(0.02, 1.50, 0.002)
    Yp, Np, W = pack(Ns_list, np.asarray(Y)[None], max(len(x) for x in Ns_list))
    c0 = float(fit_c_boot(Yp, Np, W, g)[0])
    fg = np.arange(max(0.01, c0 - 0.01), c0 + 0.01, 0.0001)
    return float(fit_c_boot(Yp, Np, W, fg)[0])


# ── Crossovers located on the measured budgets only ──────────────────────────────────────
def empirical_crossing(Ns, acc, s):
    """Where the head-fine-tuning curve first reaches the source level s.

    status 'L': already at or above s at the smallest budget (N* ≤ K_min);
    status 'I': crosses between two measured budgets (interval, with a point estimate from
                linear interpolation on log N);
    status 'R': still below s at the largest budget (N* > K_max).
    """
    Ns = np.asarray(Ns, float)
    acc = np.asarray(acc, float)
    hit = np.where(acc >= s)[0]
    if len(hit) == 0:
        return dict(status='R', lo=float(Ns[-1]), hi=None, N_hat=None,
                    delta_last_pp=float(100 * (acc[-1] - s)))
    j = int(hit[0])
    if j == 0:
        return dict(status='L', lo=None, hi=float(Ns[0]), N_hat=None,
                    delta_first_pp=float(100 * (acc[0] - s)),
                    delta_last_pp=float(100 * (acc[-1] - s)))
    a0, a1 = float(acc[j - 1]), float(acc[j])
    n0, n1 = float(Ns[j - 1]), float(Ns[j])
    frac = 0.0 if a1 == a0 else (s - a0) / (a1 - a0)
    nhat = float(np.exp(np.log(n0) + frac * (np.log(n1) - np.log(n0))))
    return dict(status='I', lo=n0, hi=n1, N_hat=nhat, bracket=[n0, n1],
                delta_last_pp=float(100 * (acc[-1] - s)))


def crossing_from_A(A, S, Ns):
    """Crossover of the fold-averaged curve of a fold-level matrix A (folds, budgets)."""
    return empirical_crossing(Ns, A.mean(0), float(S.mean()))
