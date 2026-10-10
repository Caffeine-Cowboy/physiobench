# PhysioBench

**A reproducible multimodal physiology benchmark for contradictory signals,
missing sensors and model adaptation.**

Independent research engineering by Adrian Moreno. ECG, PPG, respiration and
motion observations share an underlying physiological process, while controlled
stresses corrupt individual sensors. Models estimate mean heart and respiratory
rates, available-sensor corruption, transform severity and prediction quality.
The primary benchmark uses independently authored synthetic equations and
contains no UCLA source, patient observations or protected weights.

This release includes executed experiments, frozen checkpoints, raw test
predictions, subject-based uncertainty estimates and a separate frozen public
BIDMC domain-transfer check. It is an initial research artifact with explicit
failure analysis. It does not establish clinical validity or novel methods.

## What has actually been executed

- Two small spectral-augmented neural models: a temporal CNN (50,699 parameters)
  and a temporal Transformer (64,763 parameters).
- Three training seeds and four matched checkpoint stages: base training,
  continued base training, supervised robustness adaptation and selective
  adaptation. The adaptation arms match optimizer steps and example budgets.
- Subject-disjoint training/tuning/calibration/test partitions; two entire
  physiological equation families excluded from all development data.
- Nine paired test conditions, including all four individual sensor-drop
  ablations; 24 checkpoints and 432 per-run evaluations.
- Observable-only signal-processing and 70-feature random-forest comparators.
- Validation-only temperature scaling, a frozen rejection threshold and
  subject-max conformal bands; 2,000-draw paired subject bootstraps.
- A public BIDMC check of all 24 frozen neural checkpoints plus the signal
  baseline, with no real-data training or model selection.

**The main finding is a tradeoff, not a uniformly successful model.** On held-out
scenario families, corruption-data adaptation reduces contradictory-pulse HR
MAE from 16.57 to 11.60 bpm for the CNN and from 16.35 to 10.13 bpm for the
Transformer relative to continued-training controls. CNN clean-family OOD HR
MAE worsens from 4.26 to 5.93 bpm. Contradiction failure rates remain high, and
the selective objective adds no consistent benefit. On BIDMC, the preselected
CNN checkpoint underperforms the signal baseline: HR MAE 12.39 versus 7.13 bpm.
These limitations are part of the released evidence.

Post-training here means **supervised signal-model adaptation and calibration**.
No language model, LLM instruction SFT, preference optimization, RLHF or
frontier-scale training was performed. The code exposes a practical training
and evaluation workflow without claiming that experience.

## GitHub artifact layout

This GitHub mirror tracks every original UTF-8 source, configuration, method
document and result JSON directly. The 24 synthetic model checkpoints and 504
synthetic prediction files are transported together in the unmodified
[original v0.1.0 release bundle](release/physiobench-v0.1.0.zip), rather than
duplicated as 528 individual Git blobs. The scientific code, results and
original evidence receipts are unchanged. Only this README and CI add the
restoration step.

Before running checks or using published checkpoints, restore the approved
artifacts locally with Python's standard library:

```bash
python scripts/restore_release_artifacts.py
python scripts/verify_artifacts.py
```

The restorer checks the original bundle SHA-256
`5bf4e66f3cfb40e9d86bdb3a678b64bf535f35a1ecd26bf0ddbdd916a4e67da5`
and the exact 528 paths and per-file hashes in
[artifacts-manifest.json](release/artifacts-manifest.json). It restores only
synthetic `.pt`/`.npz` files beneath `results/`, rejects traversal, symlinks and
unexpected entries, and refuses to replace a differing existing file. Sources,
JSON receipts and documentation are never extracted over the checkout. Rerunning
it verifies already matching files. The original artifact verifier then checks
the scientific release's complete evidence, including the restored binary files.

## Install and reproduce

The measured reference run used Python 3.12 on macOS arm64 with CPU PyTorch
and two threads. Exact runtime/test versions are pinned in
[requirements-lock.txt](requirements-lock.txt). Other platforms may differ
numerically; CPU CI checks executable behavior and receipt integrity rather
than demanding cross-platform bitwise equality.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
python scripts/restore_release_artifacts.py
python -m pytest -q
python scripts/verify_artifacts.py
python -m physiobench.experiments --config configs/experiment-study.json --output results/reproduction
python -m physiobench.baseline_experiment --config configs/experiment-study.json --output results/reproduction-baselines
python scripts/compare_reproduction.py --original results/neural-study --reproduction results/reproduction --output results/reproduction-comparison.json
```

Published artifacts stay intact when reproduction uses a new output directory.
The executed neural run took about 68 seconds on the recorded reference
machine; hardware and software are in its manifest. Training time on another
machine is not guaranteed. The smoke configuration completes a short real
training run to check wiring, not to estimate research performance:

```bash
python -m physiobench.experiments --config configs/experiment-smoke.json --output results/smoke-local
python scripts/verify_artifacts.py --results results/smoke-local
```

To generate separately stored synthetic observations and labels:

```bash
python -m physiobench.data --output data/synthetic --subjects 180 --windows 6
```

## Frozen public-data check

Install the optional external dependencies using
[requirements-external-lock.txt](requirements-external-lock.txt), then acquire
the public [BIDMC PPG and Respiration Dataset](https://physionet.org/content/bidmc/1.0.0/)
into an ignored local directory:

```bash
python -m pip install -r requirements-external-lock.txt
python -m pip install --no-deps -e '.[external]'
python -m physiobench.external_validation --download --local-data data/bidmc-public --checkpoints results/neural-study/checkpoints --report results/reproduction-bidmc.json
```

The protocol fixes three 20-second windows per recording, channel mapping,
resampling, reference extraction, estimator identities and all checkpoint/source
hashes before scoring. Each manual respiratory annotator is evaluated
separately. Recording-level uncertainty is used because distinct patient
identity cannot be established from the released recording IDs. Published
records contain aggregate errors only; public raw recordings are downloaded
locally and are not redistributed here. The dataset's attribution license is
separate from this repository's software license.

Read [the frozen external protocol](docs/external-validation-protocol.md)
before interpreting real-signal results. This is an exploratory domain-shift
experiment using ICU data, with monitor-derived HR references and manual
respiratory annotations. It makes no clinical efficacy claim.

## Evidence and methods

- [Training/adaptation study](docs/training-study.md): architecture, objective,
  controls, numerical findings and uncertainty assumptions.
- [Benchmark protocol](docs/PROTOCOL.md) and [data card](docs/DATA_CARD.md):
  independent simulator, splits, stresses and limitations.
- [Observable baseline methods](docs/BASELINES.md).
- [Neural receipt](results/neural-study/manifest.json),
  [seed aggregates](results/neural-study/aggregate_metrics.json) and
  [paired comparisons](results/neural-study/paired_subject_bootstrap.json).
- [Baseline receipt](results/baselines/manifest.json) and
  [baseline metrics](results/baselines/metrics.json).
- [Frozen BIDMC protocol](results/external-validation.protocol.json) and
  [all estimator results](results/external-validation.json).

The checksum verifier validates every recorded model/result file, the critical
executed source hashes and the external freeze. Incidental source snapshots are
distinguished from modules actually used by a pipeline. A separate reproduction
receipt records any audited source-only metadata revision; original checkpoints
and outcome files remain frozen. Checkpoint loading uses `weights_only=True`.

## Research extensions

The next scientific questions are mechanism-level rather than a larger model
by default: whether explicit modality gating prevents contradiction errors;
whether matching training missingness to motion-drop tests removes the observed
shortcut; whether clean-data retention and loss scheduling avoid adaptation
regressions; and whether these conclusions survive an independently specified
simulator and a separately held-out real dataset. Such extensions require new
development/test partitions and a fresh frozen protocol. Repeated tuning on
this release's test outcomes would invalidate their role as held-out evidence.

Software and independently generated synthetic artifacts are under the
[MIT license](LICENSE). Public dataset rights and attribution remain with their
respective providers.
