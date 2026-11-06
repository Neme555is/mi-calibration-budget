# -*- coding: utf-8 -*-
"""Preregistered test of the budget law on a fourth dataset (OpenBMI MI).

The test has three layers, frozen in this order in the study:

  A  thresholds derived from the three-dataset fit only (36 units, no OpenBMI output used):
     python scripts/09_preregistration.py layer-a
  B  OpenBMI frozen-source accuracies (forward pass of the trained backbones, no target
     labels, no head fitted), frozen before any OpenBMI head fine-tuning:
     python scripts/09_preregistration.py layer-b
  C  verdicts after the OpenBMI head fine-tuning (scripts/04_crossover.py computes it):
     python scripts/09_preregistration.py layer-c

Predictions (thresholds are recomputed from the data by layer A, never typed in):
  P1  power-law form: R² of the OpenBMI units at the frozen c ≥ R_min
  P2a OpenBMI's own exponent inside a leave-one-dataset-out interval (descriptive if > 0.3 wide)
  P2b pooled exponent moves by less than 0.03 when OpenBMI joins (48 units)
  P3  corr(headroom, b) ≥ 0.85 within OpenBMI
  P4a |corr(source accuracy, headroom)| < 0.5 within OpenBMI
  P4b corr(source accuracy, headroom) ≤ −0.4 over the 48 units
  P5  censoring of the crossovers (L/I/R), descriptive
  P6a between-subject SD in [12, 16] pp; P6b curve span in [6.0, 10.3] pp and above the MDE
  P7  dataset-level regression band, descriptive
Core five: P1, P2b, P3, P4a, P4b.

Inputs: budget_law_3ds.json (scripts/03_budget_curves.py --datasets 2a cho2017 physionet),
evaluation_granularity.json, the OpenBMI features, foldlevel_curves.npz.
Outputs: preregistration_layerA.json, preregistration_layerB.json,
preregistration_verification.json
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy import stats

from calibration_budget import config
from calibration_budget.curves import CurveSet, crossing_from_A, fit_shared_c, r2

DS3 = ['2a', 'cho2017', 'physionet']
LAW3 = 'budget_law_3ds.json'
OUT_A, OUT_B, OUT_C = ('preregistration_layerA.json', 'preregistration_layerB.json',
                       'preregistration_verification.json')
OB = dict(dataset='OpenBMI MI (Lee et al., 2019)', n_subj=54, n_class=2, n_ch=62, chance=0.5,
          fs_raw=1000, fs_used=128,
          calib='day-1 offline block, 50 trials per class', eval='day-2 offline block, 50 trials per class',
          grid_official=[5, 20, 50], grid_dense=[5, 10, 20, 35, 50], cap_per_class=50,
          n_eval_trials=100, excluded='online-feedback blocks (not used in any layer)')
N_FOLD, N_EVAL = 54, 100
K_OFFICIAL, K_DENSE = OB['grid_official'], OB['grid_dense']
IDX_OFF = [K_DENSE.index(k) for k in K_OFFICIAL]
N_BOOT, BOOT_SEED = 2000, 20260915
C_GRID_TOP = 1.49
CORE_FIVE = ['P1', 'P2b', 'P3', 'P4a', 'P4b']


def load_law3():
    D = config.load_json(LAW3)
    assert set(D['dataset_meta']) == set(DS3), 'layer A must use the three-dataset fit'
    sub = {k: dict(N=v['N'], acc=v['acc']) for k, v in D['per_unit'].items()}
    return D, CurveSet(sub)


# ══ Layer A ═══════════════════════════════════════════════════════════════════════════
def layer_a():
    D, CS = load_law3()
    PU = D['per_unit']
    idx = {d: np.array([i for i, k in enumerate(CS.keys) if k.split('|')[0] == d]) for d in DS3}
    c_pool = fit_shared_c(CS)
    per_ds_c = {d: fit_shared_c(CS, idx=idx[d]) for d in DS3}
    der = {}

    # P1: R_min = floor(min out-of-fold R² of a three-fold leave-one-dataset-out / 0.005) × 0.005
    oof = {}
    for d in DS3:
        tr = np.concatenate([idx[x] for x in DS3 if x != d])
        c_tr = fit_shared_c(CS, idx=tr)
        oof[d] = dict(c_train=c_tr, c_own=per_ds_c[d], delta_c=per_ds_c[d] - c_tr,
                      r2_out_of_fold=r2(CS, c_tr, idx[d]))
    r2_vals = [v['r2_out_of_fold'] for v in oof.values()]
    R_MIN = float(np.floor(min(r2_vals) / 0.005) * 0.005)
    der['P1'] = dict(rule='R_min = floor(min out-of-fold R² / 0.005) × 0.005', lodo=oof,
                     r2_values=r2_vals, r_min_raw=float(min(r2_vals)), R_min=R_MIN)

    # P2a: c_pooled ± t_.975(2) · SD(Δc) · sqrt(1 + 1/3)
    dl = np.array([v['delta_c'] for v in oof.values()])
    sd_d = float(dl.std(ddof=1))
    t_crit = float(stats.t.ppf(0.975, len(dl) - 1))
    half = t_crit * sd_d * float(np.sqrt(1 + 1 / len(dl)))
    lo, hi = c_pool - half, c_pool + half
    der['P2a'] = dict(rule='c_pooled ± t_.975(2)·SD(Δc)·sqrt(1+1/3), Δc from the LODO fits',
                      delta_c=[float(x) for x in dl], sd_delta=sd_d, t_crit=t_crit, half_width=half,
                      point=float(c_pool), lo=float(lo), hi=float(hi), width=float(hi - lo),
                      max_abs_delta_c=float(np.abs(dl).max()), descriptive_trigger=bool(hi - lo > 0.3),
                      bootstrap_ci_for_contrast=[float(x) for x in D['pooled']['ci95']])

    # P2b: sensitivity of the pooled exponent to OpenBMI's exponent, mapped with synthetic
    # OpenBMI curves shaped like the Cho2017 units (budgets 5, 20, 50).
    cho = [k for k in CS.keys if k.split('|')[0] == 'cho2017']
    a_t = np.array([PU[k]['a'] for k in cho])
    b_t = np.array([PU[k]['b'] for k in cho])
    base = {k: dict(N=v['N'], acc=v['acc']) for k, v in PU.items()}
    Ns_ob = np.array(K_OFFICIAL, float)

    def synth(c_ob):
        syn = {'openbmi|' + k.split('|')[1]: dict(N=list(K_OFFICIAL),
                                                   acc=[float(a_t[i] - b_t[i] * N ** (-c_ob)) for N in Ns_ob])
               for i, k in enumerate(cho)}
        return CurveSet({**base, **syn})

    mapped = [(float(c), float(fit_shared_c(synth(c)) - c_pool))
              for c in np.round(np.arange(0.04, 0.62, 0.01), 4)]
    ok = [c for c, dv in mapped if abs(dv) < 0.03]
    der['P2b'] = dict(rule='drift of the 48-unit pooled c when 12 synthetic OpenBMI curves are added',
                      threshold=0.03, mapping=[[c, dv] for c, dv in mapped],
                      c_ob_range_within_threshold=[float(min(ok)), float(max(ok))] if ok else None,
                      upper_bound=float(max(ok)) if ok else None)

    # P3 / P4a / P4b and their power
    ks = list(CS.keys)
    src = np.array([PU[k]['src_acc'] for k in ks])
    head = np.array([PU[k]['a'] - PU[k]['src_acc'] for k in ks])
    b_ = np.array([PU[k]['b'] for k in ks])
    ds_of = np.array([k.split('|')[0] for k in ks])

    def corr(x, y, m=None):
        return float(np.corrcoef(x, y)[0, 1]) if m is None else float(np.corrcoef(x[m], y[m])[0, 1])

    within = {d: dict(src_headroom=corr(src, head, ds_of == d), headroom_b=corr(head, b_, ds_of == d))
              for d in DS3}
    se12 = 1 / np.sqrt(12 - 3)
    se48 = 1 / np.sqrt(4 * 12 - 3)

    def power_lower(thr, rho, se):
        return float(1 - stats.norm.cdf((np.arctanh(thr) - np.arctanh(rho)) / se))

    der['P3'] = dict(rule='corr(headroom_e, b_e) ≥ 0.85 (one-sided), n = 12',
                     observed_within={d: within[d]['headroom_b'] for d in DS3}, se_z=se12,
                     power_at={f'rho={r}': power_lower(0.85, float(r), se12)
                               for r in ('0.92', '0.88', '0.80')})
    der['P4a'] = dict(rule='|corr(src_e, headroom_e)| < 0.5 (two-sided), n = 12',
                      observed_within={d: within[d]['src_headroom'] for d in DS3}, se_z=se12,
                      alpha_false_positive=float(2 * (1 - stats.norm.cdf(np.arctanh(0.5) / se12))))
    der['P4b'] = dict(rule='corr(src, headroom) over the 48 units ≤ −0.4 (one-sided)',
                      observed_pooled_36=corr(src, head), se_z_48=se48,
                      alpha_false_positive=float(1 - stats.norm.cdf(np.arctanh(0.4) / se48)))

    # P6a / P6b from the noise decomposition of the three datasets
    EG = config.load_json('evaluation_granularity.json')['decompose']
    der['P6a'] = dict(rule='[12, 16] pp: the three observed values with about 1 pp margin',
                      observed={d: float(EG[d]['sd_between_pp']) for d in DS3}, lo=12.0, hi=16.0)
    ident = {}
    for d in DS3:
        m = ds_of == d
        Ns = np.array(PU[ks[int(np.where(m)[0][0])]]['N'], float)
        bracket = float(Ns.min() ** (-c_pool) - Ns.max() ** (-c_pool))
        ident[d] = dict(bbar=float(b_[m].mean()), bracket=bracket,
                        pred_span_pp=float(100 * b_[m].mean() * bracket),
                        obs_span_pp=float(EG[d]['span_pp']))
    bbar_obs = [float(b_[ds_of == d].mean()) for d in DS3]
    bracket_ob = float(K_OFFICIAL[0] ** (-c_pool) - K_OFFICIAL[-1] ** (-c_pool))
    tsum = float(stats.t.ppf(0.975, N_FOLD - 1) + stats.t.ppf(0.80, N_FOLD - 1))
    der['P6b'] = dict(identity='span = mean(b) × [K_min^(−c) − K_max^(−c)]', identity_check=ident,
                      bracket_openbmi=bracket_ob, bbar_observed_range=[min(bbar_obs), max(bbar_obs)],
                      span_pred_pp=[float(100 * min(bbar_obs) * bracket_ob),
                                    float(100 * max(bbar_obs) * bracket_ob)],
                      span_threshold=[6.0, 10.3], t_sum=tsum, n_subj=N_FOLD,
                      mde_by_sd={str(s): float(tsum * s / np.sqrt(N_FOLD)) for s in (12.0, 13.0, 14.0, 15.0, 16.0)})

    # P7: dataset-level regression of mean headroom on mean source accuracy (3 points)
    ms = np.array([float(src[ds_of == d].mean()) for d in DS3])
    mh = np.array([float(head[ds_of == d].mean()) for d in DS3])
    lr = stats.linregress(ms, mh)
    s_res = float(np.sqrt(((lr.slope * ms + lr.intercept - mh) ** 2).sum() / (len(ms) - 2)))
    band = float(stats.t.ppf(0.975, len(ms) - 2) * s_res * np.sqrt(1 + 1 / len(ms)))
    der['P7'] = dict(points={d: [float(s), float(h)] for d, s, h in zip(DS3, ms, mh)},
                     slope=float(lr.slope), intercept=float(lr.intercept), r=float(lr.rvalue),
                     resid_sd=s_res, band_half_width=band,
                     headroom_range=[float(mh.min()), float(mh.max())],
                     descriptive_trigger=bool(band > (mh.max() - mh.min())))

    preds = dict(
        P1=dict(threshold=R_MIN, type='adjudicated', comparison='R² >= threshold'),
        P2a=dict(interval=[float(lo), float(hi)], width=float(hi - lo),
                 type='descriptive' if hi - lo > 0.3 else 'adjudicated'),
        P2b=dict(threshold=0.03, type='adjudicated', comparison='|c_48 − c_36| < threshold',
                 equivalent_upper_bound=der['P2b']['upper_bound']),
        P3=dict(threshold=0.85, type='adjudicated', comparison='r >= threshold'),
        P4a=dict(threshold=0.5, type='adjudicated', comparison='|r| < threshold'),
        P4b=dict(threshold=-0.4, type='adjudicated', comparison='r <= threshold'),
        P5=dict(type='descriptive', note='rule declared in layer B, values reported in layer C'),
        P6a=dict(interval=[12.0, 16.0], type='auxiliary'),
        P6b=dict(interval=[6.0, 10.3], predicted_span_pp=der['P6b']['span_pred_pp'], type='auxiliary'),
        P7=dict(type='descriptive'))
    rule = dict(core_five=CORE_FIVE, all_pass='the invariants reproduce on OpenBMI',
                any_fail='partially falsified (report observed value, threshold and distance)',
                auxiliary=['P6a', 'P6b'], descriptive=['P2a', 'P5', 'P7'],
                non_adjudicable=f'if OpenBMI\'s exponent reaches the grid limit (c ≥ {C_GRID_TOP}), '
                                'P2b and the overall verdict are non-adjudicable')
    out = dict(meta=dict(layer='A', inputs=[LAW3, 'evaluation_granularity.json'], openbmi=OB,
                         baseline=dict(c_pooled_36=float(c_pool), per_dataset_c=per_ds_c,
                                       r2_36=float(r2(CS, c_pool)), n_cells=len(CS.keys))),
               predictions=preds, derivations=der, verdict_rule=rule)
    path = config.save_json(OUT_A, out)
    print(f'  c_36 = {c_pool:.4f}  R_min = {R_MIN:.3f}  P2a = [{lo:.3f}, {hi:.3f}]  '
          f'P2b upper bound = {der["P2b"]["upper_bound"]}')
    print(f'[saved] {path}')


# ══ Layer B ═══════════════════════════════════════════════════════════════════════════
def src_per_fold(ds, backbone):
    vals = []
    for fold in range(config.N_SUBJECTS[ds]):
        with np.load(os.path.join(config.features_dir(ds), backbone, f'fold{fold}.npz')) as f:
            vals.append(float((f['eval_proba_rerun'].astype(np.float64).argmax(1)
                               == f['eval_y'].astype(np.int64)).mean()))
    return np.array(vals)


def layer_b():
    D, _ = load_law3()
    eq = {}
    for ds in ('cho2017', 'physionet'):                     # same definition as the fit's src_acc
        worst = max(abs(float(src_per_fold(ds, bk).mean()) - float(D['per_unit'][f'{ds}|{bk}']['src_acc']))
                    for bk in config.BACKBONES)
        eq[ds] = dict(max_abs_diff=worst, passed=bool(worst < 1e-12))
    src = {}
    for bk in config.BACKBONES:
        v = src_per_fold('openbmi', bk)
        src[f'openbmi|{bk}'] = dict(backbone=bk, n_folds=int(len(v)), src_acc=float(v.mean()),
                                    sd_across_folds=float(v.std(ddof=1)),
                                    per_fold=[round(float(x), 6) for x in v])
    ob_mean = float(np.mean([v['src_acc'] for v in src.values()]))
    ref = {d: float(np.mean([v['src_acc'] for k, v in D['per_unit'].items() if k.startswith(d + '|')]))
           for d in DS3}
    out = dict(meta=dict(layer='B', uses='eval_proba_rerun and eval_y of the OpenBMI features only'),
               source_acc=src, openbmi_mean_src=ob_mean,
               cross_dataset=dict(existing=ref, delta_vs={d: ob_mean - ref[d] for d in DS3}),
               equivalence_check=eq)
    path = config.save_json(OUT_B, out)
    print(f'  OpenBMI mean source accuracy = {ob_mean:.4f}   equivalence check: '
          + ', '.join(f'{d} {"ok" if v["passed"] else "FAILED"}' for d, v in eq.items()))
    print(f'[saved] {path}')


# ══ Layer C ═══════════════════════════════════════════════════════════════════════════
def decompose(A_off, n_eval=N_EVAL):
    E, nf, nK = A_off.shape
    tot, bino = [], []
    for e in range(E):
        for j in range(nK):
            a = A_off[e, :, j]
            tot.append(float(a.var(ddof=1)))
            bino.append(float(np.mean(a * (1.0 - a) / n_eval)))
    tot, bino = np.array(tot), np.array(bino)
    bet = np.maximum(tot - bino, 0.0)
    curve = A_off.mean((0, 1))
    return dict(sd_between_pp=float(100 * np.sqrt(bet.mean())), sd_fold_pp=float(100 * np.sqrt(tot.mean())),
                sd_bino_pp=float(100 * np.sqrt(bino.mean())), binom_share=float(bino.sum() / tot.sum()),
                span_pp=float(100 * (curve.max() - curve.min())), curve=[float(x) for x in curve])


def p5_censoring(store):
    rng = np.random.default_rng(BOOT_SEED)
    per_cell, boot_idx = {}, {}
    for bk in config.BACKBONES:
        A, S = store[bk]['A'], store[bk]['S']
        point = crossing_from_A(A, S, K_DENSE)
        bi = rng.integers(0, N_FOLD, size=(N_BOOT, N_FOLD))
        boot_idx[bk] = bi
        Y = A[bi].mean(1)
        hit = Y >= S[bi].mean(1)[:, None]
        j = np.argmax(hit, axis=1)
        st = np.where(~hit.any(1), 2, np.where(j == 0, 0, 1))
        per_cell[bk] = dict(status=point['status'], lo=point['lo'], hi=point['hi'], N_hat=point['N_hat'],
                            delta_first_pp=point.get('delta_first_pp'),
                            delta_last_pp=point.get('delta_last_pp'),
                            p_L=float((st == 0).mean()), p_I=float((st == 1).mean()),
                            p_R=float((st == 2).mean()))
    st = [per_cell[bk]['status'] for bk in config.BACKBONES]
    Ymean = np.array([store[bk]['A'].mean(0) for bk in config.BACKBONES])
    Smean = np.array([store[bk]['S'].mean() for bk in config.BACKBONES])
    E = len(config.BACKBONES)
    frac = [float(np.mean([(Ymean[i, :j + 1] >= Smean[i]).any() for i in range(E)]))
            for j in range(len(K_DENSE))]
    Yb = np.array([store[bk]['A'][boot_idx[bk]].mean(1) for bk in config.BACKBONES])
    Sb = np.array([store[bk]['S'][boot_idx[bk]].mean(1) for bk in config.BACKBONES])
    los, his = [], []
    for j in range(len(K_DENSE)):
        fb = np.mean([(Yb[i, :, :j + 1] >= Sb[i][:, None]).any(1) for i in range(E)], axis=0)
        los.append(float(np.percentile(fb, 2.5)))
        his.append(float(np.percentile(fb, 97.5)))
    return dict(per_cell=per_cell, n_L=st.count('L'), n_I=st.count('I'), n_R=st.count('R'),
                K=K_DENSE, frac_crossed=frac, lo=los, hi=his, n_boot=N_BOOT)


def p7_band(PU, ob_point):
    src3 = np.array([float(np.mean([v['src_acc'] for k, v in PU.items() if k.startswith(d + '|')])) for d in DS3])
    head3 = np.array([float(np.mean([v['a'] - v['src_acc'] for k, v in PU.items() if k.startswith(d + '|')]))
                      for d in DS3])
    lr = stats.linregress(src3, head3)
    s_res = float(np.sqrt(((lr.slope * src3 + lr.intercept - head3) ** 2).sum() / (len(src3) - 2)))
    band = float(stats.t.ppf(0.975, len(src3) - 2) * s_res * np.sqrt(1 + 1 / len(src3)))
    pred = float(lr.slope * ob_point[0] + lr.intercept)
    return dict(points={d: [float(s), float(h)] for d, s, h in zip(DS3, src3, head3)},
                slope=float(lr.slope), intercept=float(lr.intercept), band_half_width=band,
                openbmi_point=[float(ob_point[0]), float(ob_point[1])], openbmi_predicted=pred,
                residual_pp=float(100 * (ob_point[1] - pred)), inside=bool(abs(ob_point[1] - pred) <= band))


def layer_c():
    A = config.load_json(OUT_A)
    B = config.load_json(OUT_B)
    D, _ = load_law3()
    PU = D['per_unit']
    THR = A['predictions']
    c_frozen = float(A['meta']['baseline']['c_pooled_36'])
    z = np.load(config.analysis_file('foldlevel_curves.npz'))
    store = {}
    for bk in config.BACKBONES:
        assert list(z[f'openbmi|{bk}|Ks']) == K_DENSE
        store[bk] = dict(A=z[f'openbmi|{bk}|A'],
                         S=np.array(B['source_acc'][f'openbmi|{bk}']['per_fold'], float))

    def curve_sub(idx):
        return {f'openbmi|{bk}': dict(N=[K_DENSE[i] for i in idx],
                                      acc=[float(x) for x in store[bk]['A'][:, idx].mean(0)])
                for bk in config.BACKBONES}

    ob_cs = CurveSet(curve_sub(IDX_OFF))
    a_ob, b_ob, _ = ob_cs.fit(c_frozen)
    ob_src = np.array([store[bk]['S'].mean() for bk in config.BACKBONES])
    ob_head = a_ob - ob_src
    res = {}

    r2_ob = float(r2(ob_cs, c_frozen))
    pred = a_ob[:, None] - b_ob[:, None] * ob_cs.N ** (-c_frozen)
    res['P1'] = dict(observed=r2_ob, threshold=THR['P1']['threshold'],
                     distance=r2_ob - THR['P1']['threshold'],
                     verdict='PASS' if r2_ob >= THR['P1']['threshold'] else 'FAIL',
                     per_unit_rmse_pp=[float(100 * np.sqrt(((ob_cs.Y[i] - pred[i]) ** 2).mean()))
                                       for i in range(len(ob_src))],
                     max_abs_resid_pp=float(100 * np.abs(ob_cs.Y - pred).max()),
                     per_unit={bk: dict(a=float(a_ob[i]), b=float(b_ob[i]), src=float(ob_src[i]),
                                        headroom=float(ob_head[i])) for i, bk in enumerate(config.BACKBONES)})

    c_ob = float(fit_shared_c(ob_cs))
    lo, hi = THR['P2a']['interval']
    res['P2a'] = dict(observed=c_ob, interval=[lo, hi], inside=bool(lo <= c_ob <= hi),
                      type=THR['P2a']['type'], r2_at_c_ob=float(r2(ob_cs, c_ob)),
                      unidentifiable=bool(c_ob >= C_GRID_TOP))

    base = {k: dict(N=list(v['N']), acc=list(v['acc'])) for k, v in PU.items()}
    c48 = float(fit_shared_c(CurveSet({**base, **curve_sub(IDX_OFF)})))
    delta = c48 - c_frozen
    na = c_ob >= C_GRID_TOP
    res['P2b'] = dict(observed=delta, c_pooled_48=c48, c_pooled_36=c_frozen,
                      threshold=THR['P2b']['threshold'], distance=abs(delta),
                      verdict='non-adjudicable' if na else
                              ('PASS' if abs(delta) < THR['P2b']['threshold'] else 'FAIL'))

    r_hb = float(np.corrcoef(ob_head, b_ob)[0, 1])
    r_sh = float(np.corrcoef(ob_src, ob_head)[0, 1])
    res['P3'] = dict(observed=r_hb, threshold=THR['P3']['threshold'], distance=r_hb - THR['P3']['threshold'],
                     verdict='PASS' if r_hb >= THR['P3']['threshold'] else 'FAIL')
    res['P4a'] = dict(observed=r_sh, threshold=THR['P4a']['threshold'],
                      distance=abs(r_sh) - THR['P4a']['threshold'],
                      verdict='PASS' if abs(r_sh) < THR['P4a']['threshold'] else 'FAIL')
    all_src = np.array([v['src_acc'] for v in PU.values()] + list(ob_src))
    all_head = np.array([v['a'] - v['src_acc'] for v in PU.values()] + list(ob_head))
    r48 = float(np.corrcoef(all_src, all_head)[0, 1])
    res['P4b'] = dict(observed=r48, threshold=THR['P4b']['threshold'], n=len(all_src),
                      distance=THR['P4b']['threshold'] - r48,
                      verdict='PASS' if r48 <= THR['P4b']['threshold'] else 'FAIL')

    res['P5'] = p5_censoring(store)

    dec = decompose(np.stack([store[bk]['A'][:, IDX_OFF] for bk in config.BACKBONES]))
    lo6, hi6 = THR['P6a']['interval']
    inside6 = lo6 <= dec['sd_between_pp'] <= hi6
    res['P6a'] = dict(observed=dec['sd_between_pp'], interval=[lo6, hi6],
                      verdict='PASS' if inside6 else 'FAIL',
                      distance=0.0 if inside6 else min(abs(dec['sd_between_pp'] - lo6),
                                                       abs(dec['sd_between_pp'] - hi6)),
                      breakdown=dec)
    ts = float(stats.t.ppf(0.975, N_FOLD - 1) + stats.t.ppf(0.80, N_FOLD - 1))
    mde = float(ts * dec['sd_between_pp'] / 100.0 / np.sqrt(N_FOLD) * 100)
    lo6b, hi6b = THR['P6b']['interval']
    in_band = lo6b <= dec['span_pp'] <= hi6b
    res['P6b'] = dict(observed=dec['span_pp'], interval=[lo6b, hi6b], mde_pp=mde,
                      span_over_mde=bool(dec['span_pp'] > mde),
                      verdict='PASS' if (in_band and dec['span_pp'] > mde) else 'FAIL',
                      distance=0.0 if in_band else min(abs(dec['span_pp'] - lo6b), abs(dec['span_pp'] - hi6b)))
    res['P7'] = p7_band(PU, (float(ob_src.mean()), float(ob_head.mean())))

    fails = [t for t in CORE_FIVE if res[t]['verdict'] == 'FAIL']
    na_list = [t for t in CORE_FIVE if res[t]['verdict'] == 'non-adjudicable']
    rule = A['verdict_rule']
    overall = 'non-adjudicable' if na_list else (rule['any_fail'] if fails else rule['all_pass'])
    out = dict(meta=dict(layer='C', c_frozen=c_frozen, K_official=K_OFFICIAL, K_dense=K_DENSE),
               verdict=dict(overall=overall, core_five=CORE_FIVE, failed=fails, non_adjudicable=na_list),
               results=res)
    path = config.save_json(OUT_C, out)
    for t in ('P1', 'P2b', 'P3', 'P4a', 'P4b', 'P6a', 'P6b'):
        r = res[t]
        print(f'  {t:<4} observed {r["observed"]:+.4f}   {r["verdict"]}')
    print(f'  P2a  c_OpenBMI = {c_ob:.4f}  inside [{lo:.3f}, {hi:.3f}]: {res["P2a"]["inside"]}')
    print(f'  P5   L/I/R = {res["P5"]["n_L"]}/{res["P5"]["n_I"]}/{res["P5"]["n_R"]}   '
          f'P7 inside band: {res["P7"]["inside"]}')
    print(f'  overall: {overall}')
    print(f'[saved] {path}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('layer', choices=['layer-a', 'layer-b', 'layer-c', 'all'])
    a = ap.parse_args()
    for step, fn in (('layer-a', layer_a), ('layer-b', layer_b), ('layer-c', layer_c)):
        if a.layer in (step, 'all'):
            fn()


if __name__ == '__main__':
    main()
