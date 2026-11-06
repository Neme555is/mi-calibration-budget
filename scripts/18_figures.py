# -*- coding: utf-8 -*-
"""Figures 2–6 of the paper from the analysis outputs (figure 1 is a schematic), in the order below.

  fig1_frontier    budget–accuracy frontier: head fine-tuning (both EA windows), frozen source
                   models and within-subject baselines
  fig2_crossover   interval-censored crossovers and the closed-form N* with bootstrap intervals
  fig3_curves      the 48 budget curves with the pooled power-law fit
  fig4_transfer    shape parameter, headroom across forms and EA windows, P4b levels
  fig5_locus       adaptation locus at a matched parameter budget

Colors are Okabe–Ito (safe for color-vision deficiency), with markers as a second cue. All text
is at least 8 pt at the printed size: each figure is drawn at most 43 pc wide, so it is never
scaled down when placed at the full page width.

Output: results/figures/<name>.png and .pdf

  python scripts/18_figures.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from calibration_budget import config

OUT = config.FIGURES_DIR
os.makedirs(OUT, exist_ok=True)
L = config.load_json
D = L('budget_law.json')
XO = L('crossover.json')
XPC, XPD = XO['per_cell'], XO['per_dataset']
EXI = L('exponent_identifiability.json')
GDJ = L('grid_density.json')
GD, OG = GDJ['p2b_by_grid'], GDJ['openbmi_exponent']
LA = L('level_audit.json')
HRR = L('functional_forms.json')['headroom']
EAW = L('ea_window.json')
P4W, HEW = EAW['p4b'], EAW['headroom']
IA = L('interior_adaptation.json')

W2 = 7.16
TEXTWIDTH_IN = 43 / 6.0225       # IEEE two-column text width, 43 pc
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'mathtext.fontset': 'dejavusans', 'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8, 'axes.linewidth': 0.6,
    'axes.edgecolor': '#333333', 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'axes.grid': True, 'grid.alpha': 0.22,
    'grid.linewidth': 0.5, 'legend.frameon': False, 'axes.unicode_minus': True, 'savefig.dpi': 600,
    'figure.dpi': 150, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42,
})
DS = config.DATASETS
DS_NAME = {'2a': 'BCI IV-2a', 'cho2017': 'Cho2017', 'physionet': 'PhysioNet MI', 'openbmi': 'OpenBMI'}
DS_COL = {'2a': '#D55E00', 'cho2017': '#0072B2', 'physionet': '#009E73', 'openbmi': '#CC79A7'}
DS_MK = {'2a': 'o', 'cho2017': 's', 'physionet': '^', 'openbmi': 'D'}
INK, GREY, ACC, BLUE = '#1a1a1a', '#6b6b6b', '#D55E00', '#0072B2'
WIN_C = '#E69F00'
FORMS = [('pow3', 'Power law'), ('exp3', 'Exponential'), ('wei3', 'Weibull'),
         ('pow2', 'Power, c = 1/2'), ('ilog2', 'Inverse log')]
FORM_COL = {'pow3': INK, 'exp3': '#882255', 'wei3': '#E69F00', 'pow2': '#56B4E9', 'ilog2': '#117733'}
FORM_MK = {'pow3': 'o', 'exp3': 'v', 'wei3': 's', 'pow2': 'D', 'ilog2': 'P'}
FAM_COL, FAM_MK = config.FAMILY_COLOR, config.FAMILY_MARKER


def save(fig, stem):
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'{stem}.{ext}'), bbox_inches='tight', facecolor='white',
                    pad_inches=0.02)
    plt.close(fig)
    # A figure wider than the text width would be scaled down when placed, and its text with it.
    w_in = plt.imread(os.path.join(OUT, f'{stem}.png')).shape[1] / plt.rcParams['savefig.dpi']
    assert w_in <= TEXTWIDTH_IN + 0.005, f'{stem} is {w_in:.2f} in wide, more than {TEXTWIDTH_IN:.2f} in'
    print('  ->', stem)


def panel(ax, s, x=-0.02, y=1.02):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=8.5, fontweight='bold', ha='right', va='bottom')


def log_ticks(ax, ticks):
    ax.set_xscale('log')
    ax.set_xticks(list(ticks))
    ax.set_xticklabels([f'{t:g}' for t in ticks])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())


def units(ds):
    return [k for k in D['per_unit'] if k.startswith(ds + '|')]


def ea_window_curves(ds, window):
    c = L(f'ea_window_curves_{ds}_{window}.json')
    items = sorted(c['cells'].items())
    return (c['K'], np.array([np.asarray(v['acc'], float).mean(0) for _, v in items]),
            np.array([float(np.mean(v['src'])) for _, v in items]))


def fig1_frontier():
    C_FB, C_EN = '#E69F00', '#882255'        # within-subject FBCSP and EEGNet
    fig, axes = plt.subplots(1, 4, figsize=(W2, 1.95))
    for i, (ax, ds) in enumerate(zip(axes, DS)):
        f = D['within_subject'][ds]
        K = f['K']
        ax.plot(K, np.array(f['csp_lda']) * 100, marker='v', ms=3, lw=1.0, color='#8c8c8c')
        ax.plot(K, np.array(f['fbcsp']) * 100, marker='d', ms=3, lw=1.0, color=C_FB)
        ax.plot(K, np.array(f['eegnet']) * 100, marker='x', ms=3.5, lw=1.0, color=C_EN)
        om = f['units_median']
        ax.fill_between(om['K'], np.array(om['q1']) * 100, np.array(om['q3']) * 100, color=BLUE,
                        alpha=.18, lw=0)
        ax.plot(om['K'], np.array(om['mean']) * 100, marker='o', ms=3, lw=1.5, color=BLUE)
        ax.axhline(f['src'] * 100, color=ACC, ls='--', lw=1.0)
        Kc, cur, src = ea_window_curves(ds, 'calib')
        ax.plot(Kc, np.median(cur, 0) * 100, marker='o', ms=3.2, mfc='white', mec=BLUE, mew=0.9,
                lw=1.0, ls=(0, (2.2, 1.4)), color=BLUE, zorder=4)
        ax.axhline(src.mean() * 100, color=ACC, ls=(0, (1, 1.3)), lw=1.1)
        log_ticks(ax, K)
        ax.set_title(f"({'abcd'[i]}) {DS_NAME[ds]}, n = {D['dataset_meta'][ds]['n_subj']}", loc='left', pad=3)
        ax.set_xlabel('Labeled trials per class, N')
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, max(hi, f['src'] * 100 + 1.2))
    axes[0].set_ylabel('Evaluation accuracy (%)')
    blank = Line2D([], [], lw=0, label=' ')
    h = [Line2D([], [], color=BLUE, marker='o', ms=3, lw=1.5, label='Head fine-tuning, EA on all trials'),
         Line2D([], [], color=BLUE, marker='o', ms=3.2, mfc='white', mew=0.9, lw=1.0, ls=(0, (2.2, 1.4)),
                label='Head fine-tuning, EA on calibration only'), blank,
         Line2D([], [], color=ACC, ls='--', lw=1.0, label='Frozen source model, EA on all trials'),
         Line2D([], [], color=ACC, ls=(0, (1, 1.3)), lw=1.1, label='Frozen source model, EA on calibration only'),
         blank,
         Line2D([], [], color='#8c8c8c', marker='v', ms=3, lw=1.0, label='Within-subject CSP+LDA'),
         Line2D([], [], color=C_FB, marker='d', ms=3, lw=1.0, label='Within-subject FBCSP'),
         Line2D([], [], color=C_EN, marker='x', ms=3.5, lw=1.0, label='Within-subject EEGNet')]
    fig.legend(handles=h, loc='lower center', ncol=3, bbox_to_anchor=(.5, -0.25), columnspacing=1.0,
               handlelength=2.0, handletextpad=0.5)
    fig.tight_layout(w_pad=0.6)
    save(fig, 'fig1_frontier')


def fig3_curves():
    c = D['pooled']['c']
    fig, axes = plt.subplots(1, 4, figsize=(W2, 2.05))
    seen = {}
    for i, (ax, ds) in enumerate(zip(axes, DS)):
        for k in units(ds):
            v = D['per_unit'][k]
            col = FAM_COL[v['family']]
            N = np.array(v['N'], float)
            Nf = np.linspace(N.min() * .88, N.max() * 1.12, 120)
            ax.plot(Nf, (v['a'] - v['b'] * Nf ** (-c)) * 100, color=col, lw=0.8, alpha=.8)
            ax.scatter(N, np.array(v['acc']) * 100, s=9, color=col, marker=FAM_MK[v['family']], zorder=3,
                       edgecolors='white', linewidths=.3)
            seen.setdefault(v['family'], col)
        ax.axhline(np.mean([D['per_unit'][k]['src_acc'] for k in units(ds)]) * 100, color=GREY, ls='--', lw=0.9)
        log_ticks(ax, sorted({n for k in units(ds) for n in D['per_unit'][k]['N']}))
        ax.set_title(f"({'abcd'[i]}) {DS_NAME[ds]}", loc='left', pad=3)
        ax.set_xlabel('Labeled trials per class, N')
    axes[0].set_ylabel('Accuracy (%)')
    h = [Line2D([], [], color=col, marker=FAM_MK[f], ms=3.2, lw=0.9, label=f) for f, col in seen.items()]
    h.append(Line2D([], [], color=GREY, ls='--', lw=0.9, label='Frozen source model (mean)'))
    fig.legend(handles=h, loc='lower center', ncol=5, bbox_to_anchor=(.5, -0.19), columnspacing=0.65,
               handlelength=1.3, handletextpad=0.4)
    fig.tight_layout(w_pad=0.6)
    save(fig, 'fig3_curves')


def fig2_crossover():
    XMIN, XMAX = 1.6, 3000.0
    EMP, MOD = '#20313f', ACC
    fig = plt.figure(figsize=(W2, 4.05))
    # The backbone names of the top row need a wide left margin; panel (e) keeps its legend on the right.
    gs = fig.add_gridspec(2, 4, height_ratios=[1.75, 1.0], hspace=.52, wspace=1.45, left=.135, right=.995,
                          top=.93, bottom=.19)
    gs_e = fig.add_gridspec(2, 1, height_ratios=[1.75, 1.0], hspace=.52, left=.095, right=.815, top=.93,
                            bottom=.19)
    for col, ds in enumerate(DS):
        ax = fig.add_subplot(gs[0, col])
        keys = [k for k in XPC if k.startswith(ds + '|')]

        def sk(k):
            e = XPC[k]['emp_fine']
            return {'L': (0, 0.0), 'I': (1, e['N_hat'] or 0.0), 'R': (2, 0.0)}[e['status']]
        keys.sort(key=sk)
        Nf = XPD[ds]['N']
        ax.axvspan(Nf[0], Nf[-1], color='#e8f0f8', zorder=0, lw=0)
        for yi, k in enumerate(keys):
            c, e = XPC[k], XPC[k]['emp_fine']
            b = c['boot']
            if b['N_model_ci']:
                ax.plot([max(b['N_model_ci'][0], XMIN), min(b['N_model_ci'][1], XMAX)], [yi, yi], color=MOD,
                        lw=0.7, alpha=.8, zorder=3)
            if c['N_model']:
                ax.plot([min(c['N_model'], XMAX)], [yi], marker='o', ms=3.2, mfc='white', mec=MOD, mew=0.9, zorder=5)
            if e['status'] == 'L':
                ax.plot([XMIN, Nf[0]], [yi, yi], color=EMP, lw=1.6, solid_capstyle='butt', zorder=4)
                ax.plot([Nf[0]], [yi], marker='<', ms=4, color=EMP, zorder=6)
            elif e['status'] == 'I':
                ax.plot([e['lo'], e['hi']], [yi, yi], color=EMP, lw=1.6, zorder=4)
                ax.plot([e['N_hat']], [yi], marker='o', ms=3.4, color=EMP, zorder=6)
            else:
                ax.plot([Nf[-1], min(Nf[-1] * 4.5, XMAX)], [yi, yi], color=EMP, lw=1.1, ls=(0, (1.4, 1.4)), zorder=4)
                ax.plot([Nf[-1]], [yi], marker='>', ms=4, color=EMP, zorder=6)
        ax.set_xscale('log')
        ax.set_xlim(XMIN, XMAX)
        ax.set_ylim(-.8, len(keys) - .2)
        ax.set_yticks(range(len(keys)))
        ax.set_yticklabels([config.BACKBONE_LABEL[XPC[k]['backbone']] for k in keys], fontsize=8)
        ax.grid(axis='y', alpha=0)
        ax.set_xticks([10, 100, 1000])
        ax.set_xticklabels(['10', '$10^2$', '$10^3$'])
        d = XPD[ds]
        ax.set_title(f"({'abcd'[col]}) {DS_NAME[ds]}\nL/I/R = {d['n_L']}/{d['n_I']}/{d['n_R']}", loc='left',
                     pad=3, fontsize=8, linespacing=1.15)
    # One axis title for the four top panels: each panel is narrower than the title.
    fig.text(.565, gs[0, 0].get_position(fig).y0 - .058, 'Labeled trials per class, N', ha='center', va='top')
    ax = fig.add_subplot(gs_e[1, 0])
    for ds in DS:
        d = XPD[ds]
        N = np.array(d['N'], float)
        f = np.array(d['frac_crossed']) * 100
        ax.fill_between(N, np.array(d['lo']) * 100, np.array(d['hi']) * 100, step='post', color=DS_COL[ds],
                        alpha=.14, lw=0)
        ax.step(N, f, where='post', color=DS_COL[ds], lw=1.4, label=DS_NAME[ds])
        ax.plot(N, f, DS_MK[ds], color=DS_COL[ds], ms=3.2, zorder=5)
    log_ticks(ax, [4, 5, 10, 15, 20, 36, 50, 72])
    ax.set_xlim(3.6, 80)
    ax.set_ylim(-5, 108)
    ax.set_xlabel('Labeled trials per class, N')
    ax.set_ylabel('Backbones crossed (%)')
    ax.legend(loc='center left', bbox_to_anchor=(1.01, .5), ncol=1)
    ax.set_title('(e) Fraction of backbones crossed at budget ≤ N', loc='left', pad=3, fontsize=8)
    h = [Line2D([], [], color=EMP, lw=1.6, marker='o', ms=3.4, label='Empirical crossover (interval)'),
         Line2D([], [], color=EMP, lw=0, marker='<', ms=4, label='Crossed at the lowest budget'),
         Line2D([], [], color=EMP, lw=1.1, ls=(0, (1.4, 1.4)), marker='>', ms=4, label='Not crossed within budget'),
         Line2D([], [], color=MOD, lw=0.7, marker='o', ms=3.2, mfc='white', mew=0.9,
                label='Closed-form $N^*$ with bootstrap 95% CI'),
         plt.Rectangle((0, 0), 1, 1, color='#e8f0f8', label='Measured budget range')]
    fig.legend(handles=h, loc='lower center', ncol=3, bbox_to_anchor=(.5, -0.015), columnspacing=1.3)
    save(fig, 'fig2_crossover')


def fig4_transfer():
    fig = plt.figure(figsize=(W2, 4.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=.52, wspace=.30, left=.165, right=.99,
                          top=.955, bottom=.11)
    ax = fig.add_subplot(gs[0, 0])
    dense = {d: EXI[d]['c_shared'] for d in ('2a', 'cho2017', 'physionet')}
    dense['openbmi'] = OG['c_dense_5pt']
    rows = DS + ['pool3', 'pool4', 'pool4c']
    lab = [DS_NAME[d] for d in DS] + ['Pooled, 3 datasets', 'Pooled, 4 datasets',
                                      'Pooled, 4 datasets,\nEA on calibration only']
    y = np.arange(len(rows))[::-1]
    lo_ob, hi_ob = OG['prereg_P2a_interval']
    yob = y[rows.index('openbmi')]
    ax.fill_betweenx([yob - .32, yob + .32], lo_ob, hi_ob, color=DS_COL['openbmi'], alpha=.13, lw=0)
    ax.text(lo_ob + .005, yob - .5, 'preregistered interval for OpenBMI', fontsize=8, color=DS_COL['openbmi'],
            ha='left', va='center')
    for r, yy in zip(rows, y):
        if r == 'pool4c':
            ax.plot(P4W['calib']['refit']['c'], yy, 'o', ms=4.2, color=WIN_C, zorder=4)
            continue
        if r in DS:
            c3, c5, col = D['per_dataset'][r]['c'], dense[r], DS_COL[r]
        elif r == 'pool3':
            c3, c5, col = GD['official']['c_36'], GD['dense']['c_36'], INK
        else:
            c3, c5, col = GD['official']['c_48'], GD['dense']['c_48'], INK
        if r == 'pool4':
            ax.plot(LA['c']['ci_two_stage'], [yy, yy], color=INK, lw=2.6, alpha=.30, solid_capstyle='butt', zorder=2)
            ax.plot(LA['c']['ci_unit'], [yy, yy], color=INK, lw=1.0, zorder=3)
        ax.plot([c3, c5], [yy, yy], color='#bbbbbb', lw=0.8, zorder=1)
        ax.plot(c3, yy, 'o', ms=4.2, color=col, zorder=4)
        ax.plot(c5, yy, 'o', ms=4.2, mfc='white', mec=col, mew=1.0, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(lab, fontsize=8)
    ax.grid(axis='y', alpha=0)
    ax.set_xlim(-0.05, 0.95)
    ax.set_xticks(np.arange(0, 0.95, 0.2))
    ax.set_ylim(-.7, len(rows) - .3)
    ax.set_xlabel('Shared shape parameter c')
    ax.legend(handles=[Line2D([], [], color=INK, marker='o', ls='', ms=4.2, label='3 budget levels'),
                       Line2D([], [], color=INK, marker='o', ls='', ms=4.2, mfc='white', label='Denser grid'),
                       Line2D([], [], color=INK, lw=1.0, label='95% CI, units'),
                       Line2D([], [], color=INK, lw=2.6, alpha=.30, label='95% CI, datasets')],
              loc='upper right', fontsize=8, handlelength=1.4, borderaxespad=.1)
    panel(ax, '(a)')

    ax = fig.add_subplot(gs[0, 1])
    x = np.arange(len(DS))
    off = np.linspace(-.29, .29, len(FORMS))
    for j, (fk, fn) in enumerate(FORMS):
        pt = [HRR['bootstrap'][fk]['point'][d] for d in DS]
        ci = np.array([HRR['bootstrap'][fk]['ci95'][d] for d in DS])
        xs = x + off[j] - .024
        ax.errorbar(xs, pt, yerr=[np.array(pt) - ci[:, 0], ci[:, 1] - np.array(pt)], fmt=FORM_MK[fk], ms=3.1,
                    color=FORM_COL[fk], lw=0.8, capsize=0, elinewidth=0.75, label=fn, zorder=3)
        ax.plot(xs, pt, color=FORM_COL[fk], lw=0.5, alpha=.45, zorder=2)
        ax.plot(x + off[j] + .024, [HEW['calib'][fk]['headroom_pp'][d] for d in DS], FORM_MK[fk], ms=3.1,
                mfc='white', mec=FORM_COL[fk], mew=0.8, zorder=4)
    ax.axhline(0, color=GREY, lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([DS_NAME[d] for d in DS], fontsize=8)
    ax.grid(axis='x', alpha=0)
    ax.set_ylabel('Headroom $a_e-acc^{src}_e$ (pp)')
    h = [Line2D([], [], color=FORM_COL[fk], marker=FORM_MK[fk], ms=3.1, lw=0.8, label=fn) for fk, fn in FORMS]
    h += [Line2D([], [], color=GREY, marker='o', ms=3.1, lw=0, label='EA on all trials (95% CI)'),
          Line2D([], [], color=GREY, marker='o', ms=3.1, lw=0, mfc='white', mew=0.8, label='EA on calibration only')]
    ax.set_ylim(-3.5, 42)
    ax.legend(handles=h, loc='upper right', ncol=2, fontsize=8, handlelength=1.0, columnspacing=.6,
              handletextpad=.4, borderaxespad=.1)
    panel(ax, '(b)')

    ax = fig.add_subplot(gs[1, 0])
    xf = np.arange(len(FORMS))
    ax.axhspan(-1.05, -0.4, color='#dff0e3', lw=0, zorder=0)
    ax.axhline(-0.4, color='#3f8f5f', ls='--', lw=0.8)
    ax.plot(xf, [HRR['structure'][fk]['p4b'] for fk, _ in FORMS], 'o', color=INK, ms=4, zorder=4,
            label='48 units (as registered)')
    ax.plot(xf, [LA['p4b']['by_form'][fk]['r_4'] for fk, _ in FORMS], 's', color='#8a8a8a', ms=3.8, zorder=4,
            label='4 dataset means')
    ax.legend(loc='upper left', fontsize=8, handlelength=1.0, borderaxespad=.2)
    ax.set_xticks(xf)
    ax.set_xticklabels([fn for _, fn in FORMS], rotation=25, ha='right', fontsize=8)
    ax.set_ylim(-1.05, 0.2)
    ax.grid(axis='x', alpha=0)
    ax.set_ylabel('Between-dataset r(source, headroom)', fontsize=8)
    panel(ax, '(c)', y=1.03)

    sub = gs[1, 1].subgridspec(1, 2, width_ratios=[1, 2.7], wspace=0.07)
    axL = fig.add_subplot(sub[0, 0])
    axR = fig.add_subplot(sub[0, 1], sharey=axL)
    F, C = P4W['full']['refit'], P4W['calib']['refit']
    tag = {'2a': ('BCI IV-2a', 'right', 0.0, -4.2), 'cho2017': ('Cho2017', 'left', 0.25, 0.9),
           'physionet': ('PhysioNet', 'right', -0.25, 0.9), 'openbmi': ('OpenBMI', 'right', -0.2, 0.9)}
    for axx, dsl, xl in ((axL, ['2a'], (44.3, 49.7)), (axR, ['cho2017', 'physionet', 'openbmi'], (65.4, 75.2))):
        for ds in dsl:
            s0, h0 = F['ds_src_pp'][ds], F['ds_head_pp'][ds]
            s1, h1 = C['ds_src_pp'][ds], C['ds_head_pp'][ds]
            axx.annotate('', xy=(s1, h1), xytext=(s0, h0),
                         arrowprops=dict(arrowstyle='-|>', color=DS_COL[ds], lw=0.9, shrinkA=2.5, shrinkB=2.5,
                                         mutation_scale=7))
            axx.plot(s0, h0, DS_MK[ds], ms=4.6, color=DS_COL[ds], zorder=4)
            axx.plot(s1, h1, DS_MK[ds], ms=4.6, mfc='white', mec=DS_COL[ds], mew=1.0, zorder=4)
            t, ha, dx, dy = tag[ds]
            axx.text(s0 + dx, h0 + dy, t, fontsize=8, color=INK, ha=ha, va='bottom')
        axx.set_xlim(*xl)
        axx.grid(axis='x', alpha=0)
    axL.set_ylim(0, 24.5)
    axL.set_ylabel('Headroom (pp)')
    axR.tick_params(labelleft=False, left=False)
    axR.spines['left'].set_visible(False)
    axL.plot([0.95, 1.05], [-.03, .03], transform=axL.transAxes, color='#333333', lw=0.6, clip_on=False)
    axR.plot([-0.02, 0.02], [-.03, .03], transform=axR.transAxes, color='#333333', lw=0.6, clip_on=False)
    axR.set_xlabel('Frozen source accuracy (%)')
    axR.xaxis.set_label_coords(0.33, -0.16)
    axR.legend(handles=[Line2D([], [], color=GREY, marker='o', ls='', ms=4.4, label='EA on all trials'),
                        Line2D([], [], color=GREY, marker='o', ls='', ms=4.4, mfc='white', mew=1.0,
                               label='EA on calibration only')],
               loc='upper right', fontsize=8, handlelength=1.0, borderaxespad=.1,
               title=f"Spearman ρ, 4 datasets: {F['rho4']:+.2f} → {C['rho4']:+.2f}".replace('-', '−'),
               title_fontsize=8)
    panel(axL, '(d)', x=-0.06, y=1.03)
    save(fig, 'fig4_transfer')


def fig5_locus():
    LOC, MT, LSC = IA['locus'], IA['match_table'], IA['scope']
    BK = [('conformer', 'o'), ('ctnet', 's'), ('eegnet', '^'), ('mu_cnn', 'v'), ('shallowconvnet', 'D'),
          ('deepconvnet', 'P')]
    col = {bk: FAM_COL[config.FAMILY[bk]] for bk, _ in BK}
    HEAD_C, INT_C = ACC, BLUE
    fig = plt.figure(figsize=(W2, 3.7))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.92], hspace=1.05, wspace=0.10, left=0.085, right=0.995,
                          top=0.93, bottom=0.09)
    axes = [fig.add_subplot(gs[0, 0])]
    axes += [fig.add_subplot(gs[0, i], sharey=axes[0]) for i in range(1, 4)]
    for i, (ax, ds) in enumerate(zip(axes, DS)):
        cells = LOC[ds]['cells']
        ks = sorted({c['K'] for c in cells})
        ax.axhline(0, color=GREY, lw=0.7, zorder=1)
        for bk, mk in BK:
            cc = sorted((c for c in cells if c['backbone'] == bk), key=lambda c: c['K'])
            mism = MT[f'{ds}|{bk}']['match'] != 'within_20pct'
            ax.plot([c['K'] for c in cc], [c['diff_pp'] for c in cc], ls='--' if mism else '-', color=col[bk],
                    lw=1.0, zorder=2)
            for c in cc:
                ax.plot(c['K'], c['diff_pp'], mk, ms=3.8, color=col[bk], zorder=3,
                        mfc=col[bk] if c['p'] < 0.05 else 'white', mec=col[bk], mew=0.9)
        log_ticks(ax, ks)
        ax.set_xlim(ks[0] / 1.35, ks[-1] * 1.35)
        ax.set_title(DS_NAME[ds], fontsize=8, pad=3)
        ax.grid(axis='x', alpha=0)
        if i == 0:
            ax.set_ylabel('Interior − head (pp)')
        else:
            ax.tick_params(labelleft=False)
        ax.set_xlabel('Labeled trials per class')
        panel(ax, f'({"abcd"[i]})', y=1.03)
    h = [Line2D([], [], color=col[bk], marker=mk, ms=3.8, lw=1.0, label=config.BACKBONE_LABEL[bk]) for bk, mk in BK]
    h += [Line2D([], [], color=GREY, marker='o', ms=3.8, lw=0, mfc=GREY, label='p < 0.05'),
          Line2D([], [], color=GREY, ls='--', lw=1.0, label='Mismatch, biased to adapter')]
    fig.legend(handles=h, loc='center', ncol=4, fontsize=8, frameon=False, handlelength=1.6, columnspacing=1.2,
               bbox_to_anchor=(0.54, 0.475))
    sub = gs[1, :].subgridspec(1, 2, width_ratios=[1.75, 1.0], wspace=0.22)
    ax = fig.add_subplot(sub[0, 0])
    x = np.arange(len(DS))
    for arm, sgn, cc, fill in (('head', -1, HEAD_C, True), ('adapter', 1, INT_C, False)):
        offs = sgn * np.linspace(0.06, 0.34, len(FORMS))[::sgn]
        for j, (fk, _) in enumerate(FORMS):
            ax.plot(x + offs[j], [LSC['per_ds'][ds][arm]['headroom_pp'][fk] for ds in DS], FORM_MK[fk], ms=3.4,
                    color=cc, mfc=cc if fill else 'white', mec=cc, mew=0.9, lw=0, zorder=3)
    for xx in x[:-1] + 0.5:
        ax.axvline(xx, color='#dddddd', lw=0.6, zorder=0)
    ax.axhline(0, color=GREY, lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([DS_NAME[d] for d in DS], fontsize=8)
    ax.set_xlim(-0.5, len(DS) - 0.5)
    ax.grid(axis='x', alpha=0)
    ax.set_ylabel('Headroom (pp), 6 backbones')
    hh = [Line2D([], [], color=GREY, marker=FORM_MK[fk], ms=3.4, lw=0, label=fn) for fk, fn in FORMS]
    hh += [Line2D([], [], color=HEAD_C, marker='o', ms=3.4, lw=0, label='Head fine-tuning'),
           Line2D([], [], color=INT_C, marker='o', ms=3.4, lw=0, mfc='white', mew=0.9, label='Interior adaptation')]
    ax.legend(handles=hh, loc='upper right', ncol=2, fontsize=8, handlelength=1.0, columnspacing=.8,
              borderaxespad=.1)
    panel(ax, '(e)', y=1.03)
    ax = fig.add_subplot(sub[0, 1])
    w = 0.36
    for arm, dx, cc in (('head', -w / 2, HEAD_C), ('adapter', w / 2, INT_C)):
        ax.bar(x + dx, [LSC['per_ds'][ds][arm]['n_turned'] for ds in DS], width=w, color=cc,
               alpha=0.85 if arm == 'head' else 0.35, edgecolor=cc, lw=0.8,
               label='Head fine-tuning' if arm == 'head' else 'Interior adaptation')
    ax.set_xticks(x)
    ax.set_xticklabels(['IV-2a', 'Cho2017', 'PhysioNet', 'OpenBMI'], fontsize=8)
    ax.set_yticks(range(0, 7))
    ax.set_ylim(0, 6.6)
    ax.grid(axis='x', alpha=0)
    ax.set_ylabel('Backbones above source\nat the budget cap (of 6)', fontsize=8)
    ax.legend(loc='upper right', fontsize=8, handlelength=1.0, borderaxespad=.1)
    panel(ax, '(f)', y=1.03)
    save(fig, 'fig5_locus')


if __name__ == '__main__':
    print('figures ->', OUT)
    fig1_frontier()
    fig2_crossover()
    fig3_curves()
    fig4_transfer()
    fig5_locus()
