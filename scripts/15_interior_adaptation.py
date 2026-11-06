# -*- coding: utf-8 -*-
"""Adaptation locus: fine-tune inside the backbone or at its output, with equal parameters.

Two arms share folds, labeled subsets (``default_rng(1000 + fold)``), the full-window EA,
the saved source weights and the input pipeline:

  head     frozen backbone, new classifier head (D·C + C parameters): the main analysis
  adapter  frozen classifier head at its source weights, trainable interior adapters
           (LoRA on attention/convolution weights, or unfrozen interior layers) with about
           the same number of parameters as the head of the same backbone

The adapter of each backbone is the candidate whose parameter count is closest to the head
budget (ties go to the larger candidate). Two structural mismatches give the adapter more
parameters: ShallowConvNet on two-class data (+30 %) and EEG Conformer (+21 %).
Training recipe (identical for every budget, no early stopping): Adam, lr 1e-3, no weight
decay, 200 epochs, batch 32, cross-entropy; LoRA A ~ N(0, 1/r), B = 0.

The sweep of the paper ran without a fixed torch seed (adapter initialization and mini-batch
order), so its accuracies cannot be reproduced bit for bit. This script seeds every
(fold, budget) run with seed + 1000·fold + N; reruns are repeatable and agree with the paper
up to that seed noise.

Steps
  sweep    python scripts/15_interior_adaptation.py sweep --dataset 2a
           → results/analysis/interior_adaptation_sweep.json
  analyse  python scripts/15_interior_adaptation.py analyse
           → results/analysis/interior_adaptation.json
             match_table   adapter configuration and parameter ratio per dataset × backbone
             locus         paired adapter − head differences per budget (Wilcoxon), shared-c fits
             scope         curve summaries, crossover states and headroom (five forms, at the
                           shape parameters of the main pooled fit) of both arms
The head arm reads the full-window EA features and curves of scripts/13_ea_window.py.
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from scipy import stats

from calibration_budget import config
from calibration_budget.alignment import align_subjects
from calibration_budget.curves import CurveSet, fit_shared_c
from calibration_budget.data import load_subjects
from calibration_budget.forms import FORMS, cells_with_src, fit_shared, headroom_at, pooled_fit
from calibration_budget.head import draw_subset
from calibration_budget.summary import curve_summary
from calibration_budget.training import get_fold, load_state, load_trained, prepare_inputs

SWEEP = 'interior_adaptation_sweep.json'
OUT = 'interior_adaptation.json'
SIX = config.LOCUS_BACKBONES
DS4 = config.DATASETS
ORDER = ['2a', 'cho2017', 'physionet', 'openbmi']
RECIPE = dict(optimizer='Adam', lr=1e-3, weight_decay=0.0, epochs=200, batch_size=32,
              loss='CrossEntropy', early_stopping=False, lora_init='A~N(0,1/r), B=0',
              head='frozen at source weights')
MISMATCH = {'shallowconvnet|2', 'conformer|2', 'conformer|4'}


# ── Adapters ─────────────────────────────────────────────────────────────────────────
class LoRALinear(nn.Module):
    """y = W x + b + (B A) x; only A and B are trained."""

    def __init__(self, base, r):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.A = nn.Parameter(torch.randn(r, base.in_features) / math.sqrt(r))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))

    def forward(self, x):
        return self.base(x) + torch.nn.functional.linear(x, self.B @ self.A)


class LoRAConv2d(nn.Module):
    """Low-rank update of a convolution kernel reshaped to (out, in·kh·kw)."""

    def __init__(self, base, r):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        o, i, kh, kw = base.weight.shape
        self.shape = (o, i, kh, kw)
        self.A = nn.Parameter(torch.randn(r, i * kh * kw) / math.sqrt(r))
        self.B = nn.Parameter(torch.zeros(o, r))

    def forward(self, x):
        dw = (self.B @ self.A).view(*self.shape)
        return torch.nn.functional.conv2d(x, self.base.weight + dw, self.base.bias, self.base.stride,
                                          self.base.padding, self.base.dilation, self.base.groups)


def _get(model, path):
    m = model
    for part in path.split('.'):
        m = m[int(part)] if part.isdigit() and isinstance(m, (nn.Sequential, nn.ModuleList)) else getattr(m, part)
    return m


def _set(model, path, new):
    parts = path.split('.')
    parent = _get(model, '.'.join(parts[:-1])) if len(parts) > 1 else model
    last = parts[-1]
    if last.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)):
        parent[int(last)] = new
    else:
        setattr(parent, last, new)


def _lora_cost(mod, r):
    if isinstance(mod, nn.Linear):
        return r * (mod.in_features + mod.out_features)
    o, i, kh, kw = mod.weight.shape
    return r * (o + i * kh * kw)


def _unfreeze_cost(mod):
    return sum(p.numel() for p in mod.parameters())


def plan(bk, model, head_params):
    """Adapter candidate whose parameter count is closest to the head budget."""
    names = [n for n, _ in model.named_modules()]
    cands = []
    if bk in ('ctnet', 'conformer'):
        prefix = 'trans.' if bk == 'ctnet' else 'transformer.'
        layers = sorted(n for n in names if n.startswith(prefix)
                        and (n.endswith('.queries') or n.endswith('.values')))
        vs = [p for p in layers if p.endswith('.values')]
        for nblk in range(1, len(layers) // 2 + 1):
            sel = layers[-2 * nblk:]
            for r in range(1, 5):
                cands.append((f'lora_qv_r{r}_last{nblk}blk', [('lora', p, r) for p in sel],
                              sum(_lora_cost(_get(model, p), r) for p in sel)))
        for r in range(1, 5):
            cands.append((f'lora_v_r{r}_last1blk', [('lora', vs[-1], r)], _lora_cost(_get(model, vs[-1]), r)))
    elif bk in ('eegnet', 'mu_cnn'):
        chain = ['conv_sep_point', 'conv_sep_depth', 'bn3', 'conv_temporal', 'bn1', 'conv_spatial', 'bn2']
        for k in range(1, len(chain) + 1):
            sel = chain[:k]
            cands.append(('unfreeze_' + '+'.join(sel), [('unfreeze', b) for b in sel],
                          sum(_unfreeze_cost(_get(model, b)) for b in sel)))
    elif bk == 'deepconvnet':
        for layer in ('conv2', 'conv3'):
            for r in range(1, 41):
                cands.append((f'lora_{layer}_r{r}', [('lora', layer, r)], _lora_cost(_get(model, layer), r)))
    elif bk == 'shallowconvnet':
        for r in range(1, 13):
            c = _lora_cost(_get(model, 'conv_spat'), r)
            cands.append((f'lora_convspat_r{r}', [('lora', 'conv_spat', r)], c))
            extra = _unfreeze_cost(_get(model, 'conv_time')) + _unfreeze_cost(_get(model, 'bn'))
            cands.append((f'lora_convspat_r{r}+unfreeze_convtime_bn',
                          [('lora', 'conv_spat', r), ('unfreeze', 'conv_time'), ('unfreeze', 'bn')], c + extra))
    else:
        raise ValueError(bk)
    return min(cands, key=lambda t: (abs(t[2] / head_params - 1.0), -t[2]))


def attach(model, actions):
    for p in model.parameters():
        p.requires_grad_(False)
    for kind, path, *rest in actions:
        if kind == 'lora':
            base = _get(model, path)
            _set(model, path, LoRALinear(base, rest[0]) if isinstance(base, nn.Linear) else LoRAConv2d(base, rest[0]))
        else:
            for p in _get(model, path).parameters():
                p.requires_grad_(True)
    params = [p for p in model.parameters() if p.requires_grad]
    return params, int(sum(p.numel() for p in params))


def train_eval(model, params, Xs, ys, Xe, ye, device):
    opt = torch.optim.Adam(params, lr=RECIPE['lr'], weight_decay=RECIPE['weight_decay'])
    crit = nn.CrossEntropyLoss()
    Xs_t, ys_t = torch.from_numpy(Xs).to(device), torch.from_numpy(ys).long().to(device)
    bs = min(RECIPE['batch_size'], len(ys))
    model.train()
    last = float('nan')
    for _ in range(RECIPE['epochs']):
        perm = torch.randperm(len(ys_t), device=device)
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            loss = crit(model(Xs_t[b]), ys_t[b])
            loss.backward()
            opt.step()
        last = float(loss.detach())
    model.eval()
    with torch.no_grad():
        pred = [model(torch.from_numpy(Xe[i:i + 256]).to(device)).argmax(1).cpu().numpy()
                for i in range(0, len(ye), 256)]
    return float((np.concatenate(pred) == ye).mean()), last


def head_params(ds, bk):
    with np.load(os.path.join(config.features_dir(ds), bk, 'fold0.npz')) as f:
        D = f['calib_feats'].shape[1]
    C = config.N_CLASSES[ds]
    return D * C + C


def sweep(ds, backbones, folds, budgets, seed):
    n_class = config.N_CLASSES[ds]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    recs = []
    for bk in backbones:
        hp = head_params(ds, bk)
        fmin, fmax = config.BAND[bk]
        aligned = align_subjects(load_subjects(ds, fmin=fmin, fmax=fmax))
        n_folds = len(aligned)
        print(f'[{ds}|{bk}] head parameters = {hp}', flush=True)
        for i in range(n_folds):
            if folds is not None and i not in folds:
                continue
            _, _, X_tr, _, X_val, _, Xc, yc, Xe, ye = get_fold(aligned, i)
            X_tr, X_val, Xc, Xe = prepare_inputs(bk, X_tr, X_val, Xc, Xe)
            state = load_state(os.path.join(config.CHECKPOINT_DIR, ds, bk, f'fold{i}'), device)
            for K in budgets:
                torch.manual_seed(seed + 1000 * i + K)
                model = load_trained(bk, n_class, state, X_tr, device)
                cfg_name, actions, _ = plan(bk, model, hp)
                params, actual = attach(model, actions)
                model.to(device)
                m = draw_subset(yc, K, config.SUBSET_SEED_BASE + i)
                acc, loss = train_eval(model, params, Xc[m], yc[m], Xe, ye, device)
                recs.append(dict(ds=ds, backbone=bk, adapter_cfg=cfg_name, fold=i, K=int(K), acc=acc,
                                 final_loss=loss, head_params=hp, adapter_params=actual,
                                 ratio=round(actual / hp, 4), n_shot=int(m.sum()),
                                 match=('structural_mismatch_biased_toward_adapter'
                                        if f'{bk}|{n_class}' in MISMATCH else 'within_20pct')))
                print(f'  fold {i:>3} N={K:<3} acc={acc:.4f} params={actual}/{hp} {cfg_name}', flush=True)
                del model
                if device == 'cuda':
                    torch.cuda.empty_cache()
    path = config.analysis_file(SWEEP)
    old = config.load_json(SWEEP) if os.path.exists(path) else []
    key = lambda r: (r['ds'], r['backbone'], r['adapter_cfg'], r['fold'], r['K'])
    merged = {key(r): r for r in old}
    merged.update({key(r): r for r in recs})
    config.save_json(SWEEP, sorted(merged.values(), key=key))
    print(f'[saved] {path}')


# ── Analysis ─────────────────────────────────────────────────────────────────────────
def match_table(R):
    """Adapter configuration of every dataset × backbone: the one that covers all folds and
    budgets (records of other configurations, e.g. from a trial run, are ignored)."""
    groups = {}
    for r in R:
        groups.setdefault(f"{r['ds']}|{r['backbone']}", {}).setdefault(r['adapter_cfg'], []).append(r)
    mt = {}
    for k, by_cfg in groups.items():
        r = max(by_cfg.values(), key=len)[0]
        mt[k] = {f: r[f] for f in ('head_params', 'adapter_params', 'ratio', 'adapter_cfg', 'match')}
        mt[k]['n_class'] = config.N_CLASSES[r['ds']]
    return mt


def arms(ds, R, MT):
    """head (6, folds, budgets), adapter (6, folds, budgets), source (6, folds), eval sizes."""
    c = config.load_json(f'ea_window_curves_{ds}_full.json')
    Ks = config.BUDGETS[ds]
    H = np.array([c['cells'][bk]['acc'] for bk in SIX], float)
    S = np.array([c['cells'][bk]['src'] for bk in SIX], float)
    nev = np.array(c['cells'][SIX[0]]['n_eval'], float)
    Ad = []
    for bk in SIX:
        cfg = MT[f'{ds}|{bk}']['adapter_cfg']
        got = {(x['fold'], x['K']): x['acc'] for x in R
               if x['ds'] == ds and x['backbone'] == bk and x['adapter_cfg'] == cfg}
        Ad.append([[got[(f, K)] for K in Ks] for f in range(H.shape[1])])
    return H, np.array(Ad, float), S, nev, Ks


def locus(R, MT):
    res = {}
    for ds in DS4:
        H3, Ad3, S3, _, Ks = arms(ds, R, MT)
        cells, trend, head_c, ad_c, src = [], {}, {}, {}, {}
        for e, bk in enumerate(SIX):
            H, Ad, mt = H3[e], Ad3[e], MT[f'{ds}|{bk}']
            src[bk] = float(S3[e].mean())
            head_c[bk] = {'N': Ks, 'acc': [float(v) for v in H.mean(0)]}
            ad_c[bk] = {'N': Ks, 'acc': [float(v) for v in Ad.mean(0)]}
            diffs = []
            for j, K in enumerate(Ks):
                d = Ad[:, j] - H[:, j]
                diffs.append(float(100 * d.mean()))
                cells.append(dict(backbone=bk, K=int(K), head_pp=float(100 * H[:, j].mean()),
                                  adapter_pp=float(100 * Ad[:, j].mean()), diff_pp=float(100 * d.mean()),
                                  p=float(stats.wilcoxon(d)[1]) if np.any(d) else 1.0,
                                  ratio=mt['ratio'], match=mt['match'], adapter_cfg=mt['adapter_cfg']))
            trend[bk] = dict(diff_by_K=diffs, first_minus_last=diffs[0] - diffs[-1],
                             decreasing=bool(diffs[0] > diffs[-1]))
        fits = {}
        for arm, cur in (('head', head_c), ('adapter', ad_c)):
            cs = CurveSet(cur)
            c = fit_shared_c(cs)
            a, _, _ = cs.fit(c)
            a = {k: float(v) for k, v in zip(cs.keys, a)}
            fits[arm] = dict(c=c, a=a, headroom_pp={k: 100 * (v - src[k]) for k, v in a.items()})
        res[ds] = dict(cells=cells, trend=trend, fits=fits, src=src, K=Ks, n_subj=int(H3.shape[1]))
        nd = sum(t['decreasing'] for t in trend.values())
        print(f'  {ds:<10} adapter advantage decreases with the budget for {nd}/{len(trend)} backbones')
    return res


def lir(curve, src):
    above = [a > src for a in curve]
    return 'L' if above[0] else ('I' if any(above[1:]) else 'R')


def scope(R, MT):
    PU = config.load_json('budget_law.json')['per_unit']
    by_ds = {ds: cells_with_src(PU, ds) for ds in DS4}
    theta = {name: pooled_fit(by_ds, DS4, name)['theta'] for name in FORMS}
    res = {'theta_main': {k: (None if v is None else (float(v) if np.isscalar(v) else [float(x) for x in v]))
                          for k, v in theta.items()}, 'backbones': SIX, 'per_ds': {}}
    curves = {'head': {}, 'adapter': {}}
    for ds in DS4:
        H, Ad, S, nev, Ks = arms(ds, R, MT)
        n_subj = H.shape[1]
        out = {'K': Ks, 'n_subj': int(n_subj)}
        for arm, A in (('head', H), ('adapter', Ad)):
            t2 = curve_summary(A, S, nev, n_subj)
            mean_curve, src_bk = A.mean(1), S.mean(1)
            states = [lir(mean_curve[e], src_bk[e]) for e in range(len(SIX))]
            hr = {name: float(np.mean([headroom_at(mean_curve[e], Ks, src_bk[e], name, theta[name])
                                       for e in range(len(SIX))])) for name in FORMS}
            out[arm] = dict(src_pp=t2['src_pp'], delta_pp=t2['delta_pp'], n_turned=t2['n_turned'],
                            sd_between_pp=t2['sd_between_pp'], span_pp=t2['span_pp'], mde_pp=t2['mde_pp'],
                            ratio=t2['ratio'],
                            gain_by_K_pp=[float(100 * (mean_curve[:, j] - src_bk).mean()) for j in range(len(Ks))],
                            LIR={k: states.count(k) for k in 'LIR'}, headroom_pp=hr)
            curves[arm][ds] = [(np.asarray(Ks, float), mean_curve[e], float(src_bk[e])) for e in range(len(SIX))]
        res['per_ds'][ds] = out
    order = {}
    for arm in ('head', 'adapter'):
        order[arm] = {}
        for name in FORMS:
            vals = {ds: res['per_ds'][ds][arm]['headroom_pp'][name] for ds in DS4}
            order[arm][name] = sorted(DS4, key=lambda d: -vals[d])
        for key in ('delta_pp', 'n_turned', 'sd_between_pp'):
            vals = {ds: res['per_ds'][ds][arm][key] for ds in DS4}
            order[arm][key] = sorted(DS4, key=lambda d: -vals[d])
    res['order'] = order
    res['order_ok'] = {arm: {name: order[arm][name] == ORDER for name in FORMS} for arm in order}
    res['sd_range'] = {arm: [min(res['per_ds'][d][arm]['sd_between_pp'] for d in DS4),
                             max(res['per_ds'][d][arm]['sd_between_pp'] for d in DS4)] for arm in order}
    res['pooled_c_free'] = {arm: float(fit_shared([(N, y) for ds in DS4 for (N, y, _) in curves[arm][ds]],
                                                  'pow3')[1]) for arm in ('head', 'adapter')}
    res['main_subset_headroom_pp'] = {ds: float(100 * np.mean([PU[f'{ds}|{bk}']['a'] - PU[f'{ds}|{bk}']['src_acc']
                                                               for bk in SIX])) for ds in DS4}
    for arm in order:
        print(f'  {arm:8s} headroom ordering kept under the five forms: '
              + ', '.join(f'{n} {"yes" if res["order_ok"][arm][n] else "no"}' for n in FORMS))
    return res


def analyse():
    R = config.load_json(SWEEP)
    MT = match_table(R)
    res = dict(recipe=RECIPE, match_table=MT, locus=locus(R, MT), scope=scope(R, MT))
    print(f'[saved] {config.save_json(OUT, res)}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('step', choices=['sweep', 'analyse'])
    ap.add_argument('--dataset', choices=DS4)
    ap.add_argument('--backbones', nargs='+', default=SIX, choices=SIX)
    ap.add_argument('--folds', nargs='+', type=int, default=None)
    ap.add_argument('--budgets', nargs='+', type=int, default=None)
    ap.add_argument('--seed', type=int, default=config.SEED)
    a = ap.parse_args()
    if a.step == 'analyse':
        analyse()
        return
    if not a.dataset:
        ap.error('--dataset is required for the sweep')
    sweep(a.dataset, a.backbones, set(a.folds) if a.folds else None,
          a.budgets or config.BUDGETS[a.dataset], a.seed)


if __name__ == '__main__':
    main()
