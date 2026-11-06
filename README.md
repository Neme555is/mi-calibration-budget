# Calibration budget for cross-subject motor-imagery decoding

This repository holds the code for a study of labeled calibration budgets in cross-subject
motor-imagery decoding.

The study asks how much labeled calibration data a new user needs before adapting a
cross-subject decoder pays off. Twelve backbones are trained leave-one-subject-out (LOSO) on four
motor-imagery datasets after Euclidean alignment (EA). For each held-out subject, the frozen source
model is scored first. Its classifier head is then fine-tuned with N labeled calibration trials
per class. This gives 48 budget–accuracy curves, one per backbone × dataset *unit*. They are
fitted with a power law that has a shared shape parameter:

    acc_e(N) = a_e − b_e · N^(−c)

The **headroom** a_e − acc_src,e is the gain still available above the frozen source model.
The code also locates the crossover with the source model on the measured budgets. It tests the
law on a fourth dataset under a preregistration (OpenBMI). Two design factors are varied. The EA
estimation window is the set of target trials used to estimate the EA reference covariance: all
trials or the calibration block only. The adaptation locus is either the classifier head or
interior adaptation, that is, low-rank adapters inside the backbone with as many trainable
parameters as the head. A set of controls completes the analyses.

## Contents

```
calibration_budget/          library
  config.py                  datasets, backbones, budget grids, result paths
  data/                      loaders of BCI IV-2a, Cho2017, PhysioNet MI, OpenBMI MI
  alignment.py               Euclidean alignment (two estimation windows)
  filterbank.py              filter banks, per-channel normalization
  backbones/                 the twelve backbones (two of them to download, see below)
  training.py                LOSO folds, source training, penultimate-feature extraction
  head.py                    classifier-head fine-tuning on frozen features
  curves.py                  power law with a shared shape parameter, crossover location
  forms.py                   alternative saturating forms, headroom
  summary.py                 dataset-level summary of fold-level curves
scripts/                     numbered pipeline (below)
third_party_licenses/        licenses of adapted third-party code
LICENSE                      MIT license of this repository
requirements.txt
```

## Installation

Python 3.10. Install PyTorch for your CUDA version from <https://pytorch.org>, then run:

```
pip install -r requirements.txt
```

Run the scripts from the repository root. All outputs go to `results/`; set the environment
variable `RESULTS_DIR` to use another folder.

EEG Conformer and LMDA-Net are not redistributed. EEG Conformer is under GPL-3.0 and LMDA-Net has
no license file. The steps that build these networks need them: step 1, the `features` step of
step 13, the sweep of step 15 and the retraining for step 17. Before running those steps, follow
[`calibration_budget/backbones/external/README.md`](calibration_budget/backbones/external/README.md).
The other steps only read saved features.

## Data

All datasets are read through MOABB. If the environment variable is set, the loader reads the
local files and downloads nothing. Otherwise MOABB downloads the data to its default location.

| Dataset | MOABB class | Subjects | Classes | Channels | Calibration / evaluation block | Local files |
|---|---|---|---|---|---|---|
| BCI IV-2a | `BNCI2014_001` | 9 | 4 | 22 | session 1 / session 2 (288 / 288 trials) | `BCI2A_ROOT`: `A01T.mat` … `A09E.mat` |
| Cho2017 | `Cho2017` | 52 | 2 | 64 | first 67 / remaining trials per class (134 / 66) | `CHO2017_ROOT`: `s01.mat` … `s52.mat` |
| PhysioNet MI | `PhysionetMI` | 108 (S088 excluded) | 2 | 64 | runs 4 + 8 / run 12 (≈30 / ≈15) | `PHYSIONET_ROOT`: `S001/S001R04.edf` … |
| OpenBMI MI | `Lee2019_MI` | 54 | 2 | 62 | day-1 offline block / day-2 offline block (100 / 100) | `OPENBMI_ROOT`: `sess01_subj01_EEG_MI.mat` … |

Preprocessing is the MOABB `MotorImagery` paradigm. The signals are band-passed at 4–40 Hz
(8–13 Hz for the μ-band CNN), resampled to 128 Hz and cut 0.5–4.0 s after the cue. Cho2017 is
reordered to the 64-channel layout of PhysioNet, with T9 and T10 set to zero. The online-feedback
blocks of OpenBMI are not used. Loading OpenBMI at 1000 Hz takes about an hour per band, so the
preprocessed arrays are cached in `results/cache/openbmi` (override with `OPENBMI_CACHE`).

EA is applied per subject. Source subjects always use all of their trials. The target subject
uses all trials (`full` window, as in the main analysis) or its calibration block only (`calib`
window, scripts/13).

## Pipeline

Run the steps in this order. Runtimes are for one RTX 5070 Ti and an 8-core CPU; steps 3–18
run on the CPU except where noted. In step 1, TSMNet and the two CSP pipelines run on the
CPU, so they can run alongside the GPU backbones.

| Step | Command | Output (`results/`) | Time |
|---|---|---|---|
| 1 | `python scripts/01_train_source_models.py --dataset {2a,cho2017,physionet,openbmi}` | `features/`, `checkpoints/` | ≈3 h (2a), ≈1 day (PhysioNet), ≈3 days each (Cho2017, OpenBMI) |
| 2 | `python scripts/02_within_subject_baselines.py` | `analysis/within_subject_baselines.json` | hours (EEGNet part on GPU) |
| 3 | `python scripts/03_budget_curves.py` and `python scripts/03_budget_curves.py --datasets 2a cho2017 physionet --output budget_law_3ds.json` | `analysis/budget_law.json`, `analysis/budget_law_3ds.json` | ≈10 min each |
| 4 | `python scripts/04_crossover.py` | `analysis/foldlevel_curves.npz`, `analysis/crossover.json` | ≈15 min |
| 5 | `python scripts/05_evaluation_granularity.py --metrics` | `analysis/evaluation_granularity.json` | ≈10 min |
| 6 | `python scripts/06_exponent_identifiability.py` | `analysis/exponent_identifiability.json` | ≈10 min |
| 7 | `python scripts/07_functional_forms.py` | `analysis/functional_forms.json` | ≈1 min |
| 8 | `python scripts/08_heterogeneity.py` | `analysis/heterogeneity.json` | < 1 min |
| 9 | `python scripts/09_preregistration.py all` | `analysis/preregistration_layer{A,B}.json`, `analysis/preregistration_verification.json` | < 1 min |
| 10 | `python scripts/10_level_audit.py` | `analysis/level_audit.json` | ≈3 min |
| 11 | `python scripts/11_grid_density.py` | `analysis/grid_density.json` | ≈1 min |
| 12 | `python scripts/12_extrapolation.py` | `analysis/extrapolation.json` | < 1 min |
| 13 | `python scripts/13_ea_window.py features --dataset D --window W`, then `curves` (same arguments) for every dataset D and window W, then `python scripts/13_ea_window.py report` | `ea_window/`, `analysis/ea_window*.json` | features: several hours (GPU forward passes, CSP refits); curves ≈15 min; report < 1 min |
| 14 | `python scripts/14_ordering_subsets.py` | `analysis/ordering_subsets.json` | < 1 min |
| 15 | `python scripts/15_interior_adaptation.py sweep --dataset D` for every D, then `python scripts/15_interior_adaptation.py analyse` | `analysis/interior_adaptation*.json` | sweep ≈1 day (GPU); analysis < 1 min |
| 16 | `python scripts/16_domain_adaptation.py train --dataset D` for D in 2a, cho2017, physionet, then `python scripts/16_domain_adaptation.py compare` | `domain_adaptation/`, `analysis/domain_adaptation.json` | training several days (54 LOSO runs, GPU); comparison ≈1 min |
| 17 | `python scripts/17_controls.py synthetic permutation sampling frontend` | `analysis/controls.json` | ≈20 min, plus the retraining below |
| 18 | `python scripts/18_figures.py` | `figures/` | < 1 min |

Step 17 `frontend` needs two extra trainings on BCI IV-2a. The first is an independent rerun with
EA on; the second switches EA off:

```
python scripts/01_train_source_models.py --dataset 2a --out-root results/controls/rerun \
    --backbones mu_cnn shallowconvnet ctnet deepconvnet conformer eegtcnet lmdanet fbcnet eegnet
python scripts/01_train_source_models.py --dataset 2a --out-root results/controls/noea --no-ea \
    --backbones mu_cnn shallowconvnet ctnet deepconvnet scsp conformer eegtcnet lmdanet fbcnet eegnet
```

In the study the preregistration layers were frozen in order. Layer A uses the three-dataset
fit and no OpenBMI output. Layer B uses the OpenBMI source accuracies. Both were frozen before any
OpenBMI head fine-tuning (steps 3 and 4 on OpenBMI).

## Scripts and the paper

Section, table and figure numbers refer to the submitted manuscript; S1–S12 are the sections of
its supplementary material.

| Script | Paper |
|---|---|
| 01 | Section 2.3 source training; S1 |
| 02 | Section 3.1 within-subject baselines at equal budgets; figure 2 |
| 03 | Section 2.6 budget curves, shared shape parameter, headroom; sections 3.1, 3.3, 3.4; figures 2, 4; S1 (source fit), S4 |
| 04 | Sections 2.7, 3.2 crossovers; figure 3; S7 |
| 05 | Section 2.5 span and minimum detectable effect (MDE); section 3.1; S2 |
| 06, 07 | Sections 3.3, 3.4 functional forms, shape parameter, headroom across forms; figure 5; S4 |
| 08 | Sections 3.1, 4.5 variance components, reliability, cross-backbone agreement; S3, S12 |
| 09 | Sections 2.8, 3.3 preregistration; table 4; S8 |
| 10 | Sections 2.6, 3.3, 3.4 level audit (P4b with n = 4, intervals for c); figure 5; S8 |
| 11 | Section 3.3 grid density (verdict grid versus denser grids); figure 5; S4 |
| 12 | Section 3.4 cross-dataset extrapolation; S6 |
| 13 | Sections 2.1, 3.1, 3.4 EA estimation window; table 3 and its note; figures 2, 5; S1, S9 |
| 14 | Section 2.9 backbone-subset method; sections 3.4, 3.5 |
| 15 | Sections 2.4, 3.5 adaptation locus; table 2; figure 6; S10 |
| 16 | Section 4.4 relation to domain adaptation; S11 |
| 17 | Section 3.3 shared-pipeline controls (synthetic truth, shape-parameter injection, permuted labels, sampling seeds, EA front end); S5 |
| 18 | Figures 2–6 (figure 1 is a schematic); the output files are named `fig1_frontier` to `fig5_locus` |

## Backbones

| Key | Backbone | Family | Source | Features for the head |
|---|---|---|---|---|
| `mu_cnn` | μ-band CNN: EEGNet-8,2 on 8–13 Hz | narrow-band μ power | Lawhern et al. 2018, own implementation | 224 |
| `scsp` | spectral CSP: one-vs-rest CSP in μ, low β, high β, μ+β bands, MI selection, LDA | spectral–spatial CSP | Ang et al. 2012 (four bands), own implementation | 24 |
| `fbcsp` | FBCSP: nine 4-Hz bands, MNE CSP, MI selection, LDA | filter-bank band power | Ang et al. 2012; MNE, MOABB | 24 |
| `fbcnet` | FBCNet | filter-bank band power | [ravikiran-mane/FBCNet](https://github.com/ravikiran-mane/FBCNet), MIT | 1152 |
| `shallowconvnet` | ShallowConvNet | shallow CNN | Schirrmeister et al. 2017, own implementation | 1000 |
| `deepconvnet` | DeepConvNet | deep CNN | Schirrmeister et al. 2017, own implementation | 1200 |
| `ctnet` | CTNet | CNN + attention | [snailpt/CTNet](https://github.com/snailpt/CTNet), MIT | 112 |
| `conformer` | EEG Conformer | CNN + attention | [eeyhsong/EEG-Conformer](https://github.com/eeyhsong/EEG-Conformer), GPL-3.0, **download** | 32 |
| `eegtcnet` | EEG-TCNet | CNN + attention | PyTorch re-implementation of [Thoriri/eeg-tcnet-1](https://github.com/Thoriri/eeg-tcnet-1), Apache-2.0 | 12 |
| `lmdanet` | LMDA-Net | CNN + attention | [MiaoZhengQing/LMDA-Code](https://github.com/MiaoZhengQing/LMDA-Code), no license, **download** | 666–675 |
| `tsmnet` | TSMNet | Riemannian tangent space | [rkobler/TSMNet](https://github.com/rkobler/TSMNet), BSD-3, vendored | 210 |
| `eegnet` | EEGNet-8,2 | compact CNN | Lawhern et al. 2018, own implementation | 224 |

Source training is the same for every gradient-trained backbone:

- **Split and normalization.** 10 % of the pooled source trials form the validation split. Each
  channel is z-scored with training-split statistics.
- **Optimizer and schedule.** Adam, lr 1e-3, batch 64, at most 500 epochs, with early stopping
  after 75 epochs without a gain in validation accuracy. TSMNet uses Riemannian Adam on the CPU.
- **CSP pipelines.** Both are fitted in closed form on the training split.

The features used for head fine-tuning are the inputs of each network's last linear layer. For
each held-out subject, N calibration trials per class are drawn with `default_rng(1000 + fold)`.
A new classifier is fitted on them and scored on the evaluation block. It is L2 logistic
regression with C chosen by cross-validation, or shrinkage LDA for the spectral CSP.

## Reproducibility

**What reproduces exactly.** Given the study's saved features, the analyses of steps 3–18
reproduce the paper's numbers. This was checked against the archived outputs of the study. The
saved features are not in this repository; running the pipeline from step 1 regenerates them,
up to the run-to-run noise described below.
Most outputs are identical: the budget-law fits with every bootstrap draw, the crossovers, the
preregistration layers A–C, the variance components, the extrapolation, the EA-window curves
and P4b, the adapter comparison, the domain-adaptation comparison and the controls. The rest
differ by at most 3·10⁻¹³ (floating-point summation order), and the figures are
pixel-identical. The four data loaders, EA and the two CSP pipelines are deterministic and
reproduce the archived arrays bit for bit.

The training code was checked on the CPU, where training is deterministic. For every network
it gives weights, features and probabilities bit-identical to the code used for the paper; this
covers the backbones, the within-subject EEGNet, the interior adapters and the three
domain-adaptation methods.

**What reproduces only to within run-to-run noise.** The items below are not bit-identical. They
agree with the paper up to the noise of a single training run:

- **GPU training.** cuDNN kernels are not deterministic, so two GPU trainings of the same fold
  differ. On BCI IV-2a, single-fold source accuracies differ by a few percentage points between
  identically seeded runs.
- **Random streams.** On BCI IV-2a, Cho2017 and PhysioNet the seed is set once per backbone, and
  later folds continue that stream. A partial or resumed run therefore starts those folds from a
  different state. OpenBMI uses a seed per fold (`seed + fold`). In the study, 211 of the 648
  OpenBMI folds still come from the older per-backbone stream after an interrupted run, so they
  cannot be regenerated bit for bit.
- **Interior-adaptation sweep (step 15).** The study ran it without a fixed torch seed. The
  script now seeds every (fold, budget) run, so reruns are repeatable, but not identical to the
  published sweep.
- **Within-subject EEGNet (step 2).** In the study several datasets shared one random stream. The
  script reseeds per dataset.
- **LMDA-Net.** The upstream constructor draws its initialization in a different order from the
  copy used for the paper. The architecture is the same.
- **Within-subject baselines.** Some values changed with the current library versions; the
  original code gives the same new values when run today. 2a FBCSP at N = 5 is 39.0 % instead of
  the published 37.8 %. The Cho2017 means move by at most 0.2 pp (CSP+LDA) and 0.6 pp (FBCSP),
  although single subjects can change by up to 24 pp. All other within-subject values reproduce
  exactly, and the comparison with the frozen source models is unchanged (10 of 12 backbones
  above the best within-subject baseline on Cho2017).

## Output layout

```
results/
  features/<dataset>/<backbone>/fold{k}.npz      train/calib/eval features, labels, softmax outputs
  checkpoints/<dataset>/<backbone>/fold{k}.pt     best-validation weights (.pkl: fitted CSP pipeline)
  ea_window/<dataset>/<full|calib>/<backbone>/    target features under the two EA windows
  domain_adaptation/<dataset>/<method>_<mode>_seed<s>/
  controls/{rerun,noea}/features/2a/<backbone>/  front-end control
  analysis/*.json, *.npz                          analysis outputs
  figures/                                        figures
```

## Third-party code

- **CTNet** (MIT) and **FBCNet** (MIT): adapted. The changes are listed in each file's
  docstring.
- **EEG-TCNet** (Apache-2.0): re-implemented in PyTorch from the authors' Keras code.
- **TSMNet** (BSD-3-Clause): vendored in `calibration_budget/backbones/tsmnet_lib/`. Only its
  import paths were changed.

The license texts are in `third_party_licenses/`. EEG Conformer and LMDA-Net are not included;
see `calibration_budget/backbones/external/README.md`.

## License

The code in this repository is released under the MIT License (see `LICENSE`). Code adapted from
other projects keeps its original license, as listed under Third-party code.
