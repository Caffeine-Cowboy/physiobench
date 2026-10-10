# Protocol 0.1: controlled multimodal robustness and adaptation

This protocol specifies the target comparisons before the first full experiment.
The Git revision, configuration hash, environment, seeds, and outputs must be
recorded. This is an internal preregistered implementation protocol, not an
independent public preregistration or proof of clinical validity.

## Questions and estimands

1. How do signal processing, a feature model, and a waveform neural model compare
   on held-out synthetic subjects for duration-average HR/RR?
2. What is the paired error increase when the same held-out cases undergo missing
   modality, motion artifacts, time offset, or contradictory peripheral pulse?
3. Can development-only post-training improve calibration and selective risk
   while preserving a separately frozen test set?
4. Do adaptation gains transfer to held-out simulator families, or fail there?

Primary outputs are per-task MAE in bpm, the paired change in absolute error
between baseline and adapted models, and 90% predictive interval coverage.
Secondary outputs are RMSE, bias, Gaussian NLL, fixed-coverage selective risk,
corruption AUROC/AUPRC/Brier/ECE, and simulator-control severity MAE. Corruption
AUROC is undefined for a modality with only one target class and is reported as
null. AUPRC must be accompanied by prevalence. Clean-only corruption
discrimination cannot be interpreted because it has no positive class.

## Fixed split and leakage boundary

Default generation: 180 synthetic subjects × six 20-second windows at 50 Hz.
Seed 20261010, with one window per scenario family. Subject permutation assigns
60% train, 20% validation, and 20% test. Train/validation/ID-test contain only rest,
exertion, recovery, and respiratory shift. Irregular-pulse and weak-perfusion
windows are excluded from every development stage and evaluated on test subjects.
ID and OOD test share subjects but contain different case IDs. Subjects are the
independent inference unit. Augmented copies of a case stay in that case's split.

Feature scaling, target scaling, architecture selection, early stopping,
corruption-loss weights, calibration coefficients, uncertainty thresholds,
temperature/conformal quantiles, and adaptation updates must use train or
validation subjects only. The training implementation should divide validation
subjects into adaptation and calibration folds when both fitting and coverage
calibration use validation. No per-window random split is allowed. No feature
may contain scenario, subject ID, case ID, labels, severity, or latent states.
The released waveform feature extractor is tested against label mutation.

## Training and post-training

The training implementation records its fixed corruption augmentation mix and
intensities. Train-time stressed views may be paired with clean views for an
auxiliary consistency/reliability loss, but labels must remain separate model
inputs. A supervised waveform model jointly estimates HR/RR and observation
corruption. Any post-training adaptation claim must include its training
objective, update count, parameters updated, calibration data, and comparison
to the frozen original checkpoint. A calibration-only result must be labeled
calibration rather than reinforcement learning or preference optimization.

Report signal-processing and observable-feature comparators so a neural model
cannot look impressive simply because a synthetic frequency task is easy.
Run fixed seeds with no test-selected "best run". Report all runs, failure
conditions, and training cost; compute-constrained runs do not establish a
frontier training result. Changes after initial results are protocol amendments
and should be dated and explained.

## Evaluation and uncertainty

Stress tests transform the exact same ID/OOD cases at fixed intensity. Test
labels cannot determine model selection. Subject-cluster paired bootstrap uses
2,000 resamples with a fixed seed and candidate-minus-reference absolute-error
changes; negative values are improvements. The estimand averages each subject's
within-subject error before averaging subjects. Resample subject clusters, not
individual windows. Cluster confidence intervals quantify sampled synthetic
subjects; they do not quantify uncertainty about simulator realism. Seed
variability is a separate axis and should be reported individually or averaged
within subject before bootstrap. Multiple secondary comparisons are exploratory
and no uncorrected significance declarations should be made.

For Gaussian uncertainty, report coverage and width together at 50/80/90/95%.
NLL is meaningful only for the stated predictive distribution. Selective risk
ranks cases by uncertainty scaled by fixed task tolerances (HR 5 bpm, RR 2 bpm)
and reports 25/50/75/100% coverage. A fixed abstention rule is chosen on
development data; case acceptance must never inspect test errors. Clinical
thresholds, treatment advice, and clinical severity claims are outside scope.

## Failure analysis and publication claims

Discuss model failures under missing respiration, low perfusion amplitude,
irregular pulse, noise, discordance, and alignment seams. Publish checkpoints,
configuration, aggregate metrics, per-case synthetic predictions, plots, source,
tests, and exact reproduction commands. Keep real/public external transfer
experiments separately labeled exploratory; no external dataset belongs in model
development unless the protocol is amended. A small synthetic benchmark cannot
support claims of state-of-the-art biomedical AI or doctoral equivalence by
itself; research quality comes from valid controls, honest claims, reproducibility,
and additional independent evidence.
