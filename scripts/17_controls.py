# -*- coding: utf-8 -*-
"""Controls: is the shared exponent produced by the estimator, the sampling or the front end?

A  synthetic truth    the same head fine-tuning on synthetic subjects (shared class means,
                      subject-specific rotation; six difficulty settings) — the exponent is
                      a property of the data, not of the estimator
A2 injection          curves generated with a known exponent (0.15, 0.237, 0.40, 0.70) are
                      fitted back
B  permuted labels    real features, shuffled calibration labels (three 2a backbones): flat
                      curves, the exponent is not identifiable
C  sampling seeds     subset seeds 1000 / 3000 / 5000 on all backbones of three datasets
D  EA front end       on BCI IV-2a, ten backbones: A1 the main features (EA on), A2 an
                      independent retraining with EA on, B a retraining with EA off. Run
                      scripts/01_train_source_models.py first:
                        --dataset 2a --out-root results/controls/rerun --backbones <ten>
                        --dataset 2a --out-root results/controls/noea --no-ea --backbones <ten>
                      (the spectral CSP is deterministic and is not retrained with EA on)

  python scripts/17_controls.py synthetic permutation sampling frontend
Output: results/analysis/controls.json
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from calibration_budget import config
from calibration_budget.curves import fit_c_points, params_points, r2_points
from calibration_budget.head import head_at_N, load_fold

OUT = 'controls.json'
DS3 = ['2a', 'cho2017', 'physionet']
FRONTEND = ['mu_cnn', 'shallowconvnet', 'ctnet', 'deepconvnet', 'scsp', 'conformer', 'eegtcnet',
            'lmdanet', 'fbcnet', 'eegnet']
DETERMINISTIC = {'scsp'}
CONTROL_ROOT = config.results_path('controls')


def fit_summary(sub):
    c = fit_c_points(sub)
    vals = [p[1] for v in sub.values() for p in v]
    return dict(c=c, r2=r2_points(sub, c), n_arch=len(sub), spread_pp=float(100 * (max(vals) - min(vals))))


def synthetic_curves(n_classes=4, d=128, r=1.0, n_subj=9, n_calib=72, n_eval=72, n_draws=2, seed=0):
    """x = R_s·μ_c + ε with a subject-specific rotation R_s (a translation would be absorbed by
    the intercept). r sets the Bayes rate, d the number of head parameters to estimate."""
    rng = np.random.default_rng(seed)
    mu0 = rng.normal(size=(n_classes, d))
    mu0 = mu0 / np.linalg.norm(mu0, axis=1, keepdims=True) * r
    sub = {}
    for s in range(n_subj):
        R, _ = np.linalg.qr(rng.normal(size=(d, d)))
        mu_s = mu0 @ R.T

        def draw(n):
            y = np.repeat(np.arange(n_classes), n)
            return mu_s[y] + rng.normal(size=(len(y), d)), y

        X_cal, y_cal = draw(n_calib)
        X_eva, y_eva = draw(n_eval)
        sub[f'syn{s}'] = [(N, float(np.mean([head_at_N(X_cal, y_cal, X_eva, y_eva, N, 1000 + f * 977)
                                             for f in range(n_draws)])))
                          for N in [5, 20, 72]]
    return sub


def real_curves(datasets, seed_base=1000, permute_labels=False, seed=0, only=None, root=None,
                backbones=None, budgets=None):
    rng = np.random.default_rng(seed)
    sub = {}
    for ds in datasets:
        for bk in (backbones or config.BACKBONES):
            if only is not None and f'{ds}|{bk}' not in only:
                continue
            pts = []
            for N in (budgets or config.BUDGETS[ds]):
                accs = []
                for fold in range(config.N_SUBJECTS[ds]):
                    Xc, yc, Xe, ye, _ = load_fold(os.path.join(root or config.features_dir(ds), bk, f'fold{fold}.npz'))
                    if permute_labels:
                        yc = yc[rng.permutation(len(yc))]
                    accs.append(head_at_N(Xc, yc, Xe, ye, N, seed_base + fold, config.HEAD_KIND[bk]))
                pts.append((N, float(np.mean(accs))))
            sub[f'{ds}|{bk}'] = pts
    return sub


def synthetic():
    A = {}
    for d, r in [(64, 1.0), (128, 1.0), (128, 1.4), (256, 1.4), (256, 1.8), (512, 1.4)]:
        sub = synthetic_curves(d=d, r=r, seed=11)
        key = f'd{d}_r{r}'
        A[key] = fit_summary(sub)
        med = [float(np.mean([sub[s][i][1] for s in sub])) for i in range(3)]
        A[key].update(curves=sub, median_curve=med,
                      increments_pp=[round(100 * (med[1] - med[0]), 1), round(100 * (med[2] - med[1]), 1)])
        print(f'  synthetic {key:<10} c={A[key]["c"]:.4f}  R2={A[key]["r2"]:.4f}', flush=True)
    rng = np.random.default_rng(3)
    inj = {}
    for c_true in [0.15, 0.237, 0.40, 0.70]:
        sub = {}
        for s in range(28):
            a = 0.85 - 0.30 * rng.random()
            b = 0.10 + 0.25 * rng.random()
            sub[f'syn{s}'] = [(N, a - b * N ** (-c_true)) for N in [5, 20, 72]]
        got = fit_c_points(sub)
        inj[f'c_true_{c_true}'] = dict(c_true=c_true, c_fit=got, err=abs(got - c_true))
        print(f'  injected c={c_true:.3f} → fitted {got:.4f}', flush=True)
    return {'synthetic_truth': A, 'exponent_injection': inj}


def permutation():
    B = {}
    for bk in ('mu_cnn', 'shallowconvnet', 'ctnet'):
        key = f'2a|{bk}'
        sub = real_curves(('2a',), permute_labels=True, seed=7, only={key})
        B[key] = fit_summary(sub)
        B[key]['curves'] = sub
        print(f'  permuted labels {key:<20} curve={[round(p[1], 3) for p in sub[key]]}  c={B[key]["c"]:.4f}')
    return {'permuted_labels': B}


def sampling():
    C = {}
    for sb in [1000, 3000, 5000]:
        sub = real_curves(DS3, seed_base=sb, seed=sb)
        C[str(sb)] = fit_summary(sub)
        C[str(sb)]['curves'] = sub
        print(f'  subset seed base {sb}: c={C[str(sb)]["c"]:.4f}  R2={C[str(sb)]["r2"]:.4f}', flush=True)
    cs = np.array([v['c'] for v in C.values()])
    C['_summary'] = dict(mean=float(cs.mean()), std=float(cs.std()), min=float(cs.min()),
                         max=float(cs.max()), n_seeds=len(cs))
    return {'sampling_seeds': C}


def frontend():
    conds = {'A1_main_EA': (config.features_dir('2a'), FRONTEND),
             'A2_rerun_EA': (os.path.join(CONTROL_ROOT, 'rerun', 'features', '2a'),
                             [b for b in FRONTEND if b not in DETERMINISTIC]),
             'B_no_EA': (os.path.join(CONTROL_ROOT, 'noea', 'features', '2a'), FRONTEND)}
    sweep = {}
    for tag, (root, bks) in conds.items():
        sub = real_curves(('2a',), root=root, backbones=bks, budgets=[5, 20, 72])
        sweep[tag] = {k.split('|')[1]: {str(N): acc for N, acc in pts} for k, pts in sub.items()}
    report = {}
    for tag in conds:
        sub = {f'arch|{bk}': sorted((int(K), v) for K, v in byk.items()) for bk, byk in sweep[tag].items()}
        c = fit_c_points(sub)
        report[tag] = dict(c=c, r2=r2_points(sub, c), n_arch=len(sub), per_arch=params_points(sub, c))
    cnn = [b for b in FRONTEND if b not in DETERMINISTIC]
    for tag in conds:
        sub = {f'arch|{bk}': sorted((int(K), v) for K, v in sweep[tag][bk].items()) for bk in cnn}
        c = fit_c_points(sub)
        report[f'cnn_{tag}'] = dict(c=c, r2=r2_points(sub, c), n_arch=len(sub), per_arch=params_points(sub, c))
    run, front, worse = [], [], 0
    for bk in cnn:
        for K in ('5', '20', '72'):
            a1, a2, b = sweep['A1_main_EA'][bk][K], sweep['A2_rerun_EA'][bk][K], sweep['B_no_EA'][bk][K]
            run.append(abs(a1 - a2) * 100)
            front.append(abs(a2 - b) * 100)
            worse += b < a2
    summary = dict(mean_abs_diff_rerun_pp=float(np.mean(run)), mean_abs_diff_ea_off_pp=float(np.mean(front)),
                   cells_worse_without_ea=int(worse), n_cells=len(run),
                   dc_rerun=abs(report['cnn_A1_main_EA']['c'] - report['cnn_A2_rerun_EA']['c']),
                   dc_ea_off=abs(report['cnn_A2_rerun_EA']['c'] - report['cnn_B_no_EA']['c']))
    rerun = {}
    for bk in cnn:
        d = []
        for fold in range(config.N_SUBJECTS['2a']):
            a = np.load(os.path.join(conds['A1_main_EA'][0], bk, f'fold{fold}.npz'))
            b = np.load(os.path.join(conds['A2_rerun_EA'][0], bk, f'fold{fold}.npz'))
            d.append({k: float(np.abs(a[k].astype(float) - b[k].astype(float)).max())
                      for k in ('calib_feats', 'eval_feats', 'eval_proba_rerun')})
        rerun[bk] = dict(identical_all_folds=all(max(x['calib_feats'], x['eval_feats']) == 0.0 for x in d),
                         max_abs_proba_diff=max(x['eval_proba_rerun'] for x in d))
    print(f"  EA front end: mean |Δacc| rerun {summary['mean_abs_diff_rerun_pp']:.2f} pp, EA off "
          f"{summary['mean_abs_diff_ea_off_pp']:.2f} pp; |Δc| rerun {summary['dc_rerun']:.3f}, "
          f"EA off {summary['dc_ea_off']:.3f}")
    return {'ea_frontend': dict(sweep=sweep, report=report, summary=summary, rerun_vs_main=rerun)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('parts', nargs='+', choices=['synthetic', 'permutation', 'sampling', 'frontend'])
    a = ap.parse_args()
    path = config.analysis_file(OUT)
    res = config.load_json(OUT) if os.path.exists(path) else {}
    for part in a.parts:
        res.update({'synthetic': synthetic, 'permutation': permutation, 'sampling': sampling,
                    'frontend': frontend}[part]())
        config.save_json(OUT, res)
    print(f'[saved] {path}')


if __name__ == '__main__':
    main()
