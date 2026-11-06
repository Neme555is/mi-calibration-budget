# -*- coding: utf-8 -*-
"""Filter banks and per-channel normalization shared by the backbones.

All filters are MNE FIR filters (``fir_design='firwin'``) applied in float64 to trials that
were already band-passed to 4–40 Hz and resampled to 128 Hz.
"""
import numpy as np

SFREQ = 128

# Four bands of the spectral CSP (feature columns are concatenated in sorted band-name
# order, so the names are part of the specification).
SPECTRAL_CSP_BANDS = {
    'mu': [8, 13],
    'beta_low': [13, 20],
    'beta_high': [20, 30],
    'mu_beta': [8, 30],
}
# Nine 4-Hz bands covering 4–40 Hz (FBCSP and FBCNet).
NINE_BANDS = [[4, 8], [8, 12], [12, 16], [16, 20], [20, 24],
              [24, 28], [28, 32], [32, 36], [36, 40]]
NINE_BANDS_DICT = {f'{lo}-{hi}': [lo, hi] for lo, hi in NINE_BANDS}

FBCNET_STRIDE = 4    # FBCNet ``strideFactor``; the time axis is cropped to a multiple of it


def to_3d(X):
    """(n, 1, ch, t) → (n, ch, t); 3-D input is returned unchanged."""
    return X[:, 0, :, :] if (X.ndim == 4 and X.shape[1] == 1) else X


def filter_bank_dict(X, bands, sfreq=SFREQ):
    """(n, ch, t) → {band name: (n, ch, t) float32}."""
    from mne.filter import filter_data
    out = {}
    for name, (l_freq, h_freq) in bands.items():
        X_f = filter_data(X.astype(np.float64), sfreq, l_freq, h_freq,
                          fir_design='firwin', verbose=False)
        out[name] = X_f.astype(np.float32)
    return out


def filter_bank_stack(X, bands=NINE_BANDS, sfreq=SFREQ):
    """(n, ch, t) → (n, ch, t, n_bands) float32 (input layout of the FBCSP pipeline)."""
    import mne
    n_trials, n_channels, n_times = X.shape
    X_fb = np.zeros((n_trials, n_channels, n_times, len(bands)), dtype=np.float32)
    for b, (l_freq, h_freq) in enumerate(bands):
        X_band = mne.filter.filter_data(X.astype(np.float64), sfreq, l_freq, h_freq,
                                        fir_design='firwin', verbose=False)
        X_fb[:, :, :, b] = X_band.astype(np.float32)
    return X_fb


def crop_to_multiple(X, stride):
    """Crop the time axis to the nearest multiple of ``stride`` (449 → 448 samples)."""
    excess = X.shape[-1] % stride
    return X if excess == 0 else X[..., :-excess]


def fbcnet_input(X):
    """(n, 1, ch, t) → (n, 9, ch, t'), the nine-band input of FBCNet."""
    fb = filter_bank_dict(to_3d(X), NINE_BANDS_DICT)
    X_fb = np.stack(list(fb.values()), axis=1).astype(np.float32)
    return crop_to_multiple(X_fb, FBCNET_STRIDE)


def normalize_channels(X_tr, *others):
    """Per-channel z-score of (n, 1, ch, t) arrays with the statistics of ``X_tr``.

    The statistics come from the source training trials only and are applied unchanged to
    every other split (validation, calibration and evaluation trials of the target).
    """
    eps = 1e-8
    X_tr = X_tr.copy()
    others = [X.copy() for X in others]
    for ch in range(X_tr.shape[2]):
        mu = float(np.mean(X_tr[:, :, ch, :]))
        std = float(np.std(X_tr[:, :, ch, :]))
        denom = std + eps
        for X in (X_tr, *others):
            X[:, :, ch, :] = (X[:, :, ch, :] - mu) / denom
    return (X_tr, *others)


def normalize_bands(X_tr, *others):
    """Per-band, per-channel z-score of (n, bands, ch, t) arrays with ``X_tr`` statistics."""
    eps = 1e-8
    X_tr = X_tr.copy()
    others = [X.copy() for X in others]
    for b in range(X_tr.shape[1]):
        for ch in range(X_tr.shape[2]):
            mu = float(np.mean(X_tr[:, b, ch, :]))
            std = float(np.std(X_tr[:, b, ch, :]))
            denom = std + eps
            for X in (X_tr, *others):
                X[:, b, ch, :] = (X[:, b, ch, :] - mu) / denom
    return (X_tr, *others)
