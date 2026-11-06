# -*- coding: utf-8 -*-
"""Datasets, backbones, budget grids and result paths shared by all scripts.

Every script reads and writes below ``RESULTS_DIR`` (default: ``<repo>/results``;
override with the ``RESULTS_DIR`` environment variable)::

    results/
      features/<dataset>/<backbone>/fold{k}.npz      penultimate-layer features
      checkpoints/<dataset>/<backbone>/fold{k}.pt    CNN weights (.pkl for CSP pipelines)
      ea_window/<dataset>/<window>/<backbone>/...    features under the two EA windows
      analysis/*.json                                analysis outputs
      figures/                                       figures
"""
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.environ.get('RESULTS_DIR', os.path.join(REPO_ROOT, 'results'))


def results_path(*parts):
    return os.path.join(RESULTS_DIR, *parts)


FEATURES_DIR = results_path('features')
CHECKPOINT_DIR = results_path('checkpoints')
EA_WINDOW_DIR = results_path('ea_window')
ANALYSIS_DIR = results_path('analysis')
FIGURES_DIR = results_path('figures')


def features_dir(dataset, root=None):
    return os.path.join(root or FEATURES_DIR, dataset)


def analysis_file(name):
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    return os.path.join(ANALYSIS_DIR, name)


# ── Datasets ──────────────────────────────────────────────────────────────────
DATASETS = ['2a', 'cho2017', 'physionet', 'openbmi']
DATASET_LABEL = {'2a': 'BCI IV-2a', 'cho2017': 'Cho2017', 'physionet': 'PhysioNet MI',
                 'openbmi': 'OpenBMI MI'}
DATASET_META = {
    '2a':        dict(n_subj=9,   n_class=4, n_ch=22, chance=0.25, fs=250, calib_cap=288,
                      eval_trials=288,
                      split='cross-session (session 1 calibration / session 2 evaluation)'),
    'cho2017':   dict(n_subj=52,  n_class=2, n_ch=64, chance=0.50, fs=512, calib_cap=134,
                      eval_trials=66,
                      split='within session (first 2/3 calibration / last 1/3 evaluation)'),
    'physionet': dict(n_subj=108, n_class=2, n_ch=64, chance=0.50, fs=160, calib_cap=30,
                      eval_trials=15,
                      split='separate runs (runs 4 + 8 calibration / run 12 evaluation)'),
    'openbmi':   dict(n_subj=54,  n_class=2, n_ch=62, chance=0.50, fs=1000, calib_cap=100,
                      eval_trials=100,
                      split='cross-day (day 1 calibration / day 2 evaluation); '
                            'online-feedback blocks excluded'),
}
N_SUBJECTS = {d: DATASET_META[d]['n_subj'] for d in DATASETS}
N_CLASSES = {d: DATASET_META[d]['n_class'] for d in DATASETS}

# Labeled calibration trials per class. The last entry is the budget cap.
BUDGETS = {'2a': [5, 20, 72], 'cho2017': [5, 20, 67], 'physionet': [5, 10, 15],
           'openbmi': [5, 20, 50]}
# Denser grids used to localize crossovers; each contains the fitting grid.
DENSE_BUDGETS = {'2a': [5, 10, 20, 36, 72], 'cho2017': [5, 10, 20, 34, 67],
                 'physionet': [4, 5, 6, 8, 10, 11, 15], 'openbmi': [5, 10, 20, 35, 50]}
# Five-level grids of the grid-density analysis (three datasets).
FIVE_POINT_BUDGETS = {'2a': [5, 10, 20, 36, 72], 'cho2017': [5, 10, 20, 34, 67],
                      'physionet': [4, 6, 8, 11, 15]}

# ── Backbones ─────────────────────────────────────────────────────────────────
# Canonical order. Analyses that resample backbones index them in this order.
BACKBONES = ['mu_cnn', 'scsp', 'fbcsp', 'fbcnet', 'shallowconvnet', 'deepconvnet',
             'ctnet', 'conformer', 'eegtcnet', 'lmdanet', 'tsmnet', 'eegnet']
BACKBONE_LABEL = {
    'mu_cnn': 'μ-band CNN', 'scsp': 'Spectral CSP', 'fbcsp': 'FBCSP', 'fbcnet': 'FBCNet',
    'shallowconvnet': 'ShallowConvNet', 'deepconvnet': 'DeepConvNet', 'ctnet': 'CTNet',
    'conformer': 'EEG Conformer', 'eegtcnet': 'EEG-TCNet', 'lmdanet': 'LMDA-Net',
    'tsmnet': 'TSMNet', 'eegnet': 'EEGNet',
}
FAMILY = {
    'mu_cnn': 'Narrow-band μ power', 'scsp': 'Spectral–spatial CSP',
    'fbcsp': 'Filter-bank band power', 'fbcnet': 'Filter-bank band power',
    'shallowconvnet': 'Shallow CNN', 'deepconvnet': 'Deep CNN',
    'ctnet': 'CNN + attention', 'conformer': 'CNN + attention',
    'eegtcnet': 'CNN + attention', 'lmdanet': 'CNN + attention',
    'tsmnet': 'Riemannian tangent space', 'eegnet': 'Compact CNN',
}
FAMILY_ORDER = ['Narrow-band μ power', 'Spectral–spatial CSP', 'Filter-bank band power',
                'Shallow CNN', 'Deep CNN', 'CNN + attention', 'Riemannian tangent space',
                'Compact CNN']
# Okabe–Ito colors, safe for color-vision deficiency. Eight families are more than any palette
# can keep apart pairwise, so each family also has its own marker.
FAMILY_COLOR = {
    'Narrow-band μ power': '#D55E00', 'Spectral–spatial CSP': '#E69F00',
    'Filter-bank band power': '#882255', 'Shallow CNN': '#009E73', 'Deep CNN': '#56B4E9',
    'CNN + attention': '#0072B2', 'Riemannian tangent space': '#CC79A7',
    'Compact CNN': '#1a1a1a',
}
FAMILY_MARKER = {
    'Narrow-band μ power': 'o', 'Spectral–spatial CSP': 's', 'Filter-bank band power': '^',
    'Shallow CNN': 'v', 'Deep CNN': 'D', 'CNN + attention': 'P', 'Riemannian tangent space': 'X',
    'Compact CNN': '*',
}
# Input band (Hz): the μ-band CNN sees 8–13 Hz, every other backbone 4–40 Hz.
BAND = {b: ((8, 13) if b == 'mu_cnn' else (4, 40)) for b in BACKBONES}
# Classifier used for head fine-tuning: shrinkage LDA on the 24 selected CSP features of the
# spectral CSP, L2 logistic regression elsewhere.
HEAD_KIND = {b: ('lda_auto' if b == 'scsp' else 'lr') for b in BACKBONES}
# Nominal dimension D of the shared-b model check (budget scaled as N/D): 24 for the
# spectral CSP and 224 for every other backbone, as in the paper. It is not the actual width
# of each backbone's penultimate features (README, "Backbones").
FEATURE_DIM = {b: (24 if b == 'scsp' else 224) for b in BACKBONES}
CLOSED_FORM = ('scsp', 'fbcsp')

# The six gradient-trained backbones of the adaptation-locus experiment.
LOCUS_BACKBONES = ['conformer', 'ctnet', 'eegnet', 'mu_cnn', 'shallowconvnet', 'deepconvnet']

SEED = 42
SUBSET_SEED_BASE = 1000          # labeled subset of fold k is drawn with seed 1000 + k


# ── Analysis files ────────────────────────────────────────────────────────────────────
def save_json(name, obj):
    import json
    path = analysis_file(name)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
    return path


def load_json(name):
    import json
    with open(analysis_file(name), encoding='utf-8') as f:
        return json.load(f)
