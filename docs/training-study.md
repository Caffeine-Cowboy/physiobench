# Train, adapt, calibrate: a controlled synthetic study

This repository contains an executed neural-model study, rather than a training
plan or invented performance numbers. The scope is multimodal waveform
engineering under controlled simulation. It is not evidence of clinical
efficacy, a biophysical digital twin, a foundation-model training run, or LLM
post-training experience.

## Research questions and controls

The study asks whether exposing a trained physiological signal model to
contradictory and incomplete observations improves held-out error; whether an
explicit selective-prediction objective adds benefit beyond the same adaptation
examples; and whether confidence calibration transfers to unseen subjects and
scenario families. A development pilot informed additions of a severity head
and clustered prediction intervals. The final configuration was then fixed for
the published rerun. This is not an externally preregistered experiment; the
held-out results are exploratory evidence and need independent replication.

Four stages share an identical initial checkpoint for each architecture/seed:

1. **Base:** supervised learning on clean and motion-artifact observations.
2. **Continued base:** ordinary supervised updates using repeated clean and
   motion-artifact observations, with the same number of optimizer steps and
   training cases as both adaptation arms.
3. **Robust adaptation:** ordinary supervised updates on clean, motion,
   missing-modality, temporal-offset and contradiction examples.
4. **Selective adaptation:** the identical examples and optimizer budget as
   robust adaptation, adding an acceptance-weighted regression risk and a
   penalty for expected acceptance below 80%.

The first matched comparison isolates the **adaptation data package**, not
simply extra training time. The second isolates the **selective objective**.
The repeated continued-base collection contains three clean and two motion
views per training case; the original base collection contains one of each.
The objective comparison uses exactly identical five-condition collections.
All arms complete their allocated updates; selecting an epoch on tuning data
does not retrospectively reduce their compute budgets.

## Inputs, architecture and objective

Models see ECG, PPG, respiration and motion waveforms plus an explicit observed
mask. They do not see subject identifiers, scenario names, physiological
latents, corruption targets or severity targets at inference. A counterfactual
test changes every hidden label and identifier while preserving observations;
the predictions remain bitwise identical.

Both architectures use a disclosed 0.1–4 Hz spectral branch alongside temporal
features. Each observed channel is centered and standardized per window, while
log amplitude and observed flags are retained as auxiliary features. Hann-window
power is normalized and pooled to 80 frequency bins per modality. The CNN uses
two strided temporal convolutions. The Transformer uses 20-sample patches,
32-dimensional tokens, sinusoidal position encoding and two four-head encoder
layers. Both fuse their temporal and spectral branches into a shared 48-unit
representation. These are small spectral-augmented neural baselines, not a fair
claim of fully raw-waveform foundation modeling. Parameter counts and exact
layer choices are in the saved training histories and source.

The supervised loss combines normalized two-target Huber regression, binary
available-sensor corruption detection, squared control-severity error and a
binary correctness head. The targets use fixed affine units: `(HR−100)/40`
and `(RR−20)/10`, rather than any normalization fit to test labels. Correctness
means both absolute HR error ≤5 bpm and absolute RR error ≤3 breaths/min;
these are declared benchmark tolerances, not clinical acceptance criteria.
The correctness target is recomputed from detached current model residuals
using training labels. Corruption and severity losses exclude unobserved
sensors. Missingness itself is supplied by the observable mask and is not
claimed as learned detection. Motion truthfully measures movement and is a
negative control for sensor corruption in this simulator.

Selective adaptation uses half ordinary regression risk and half regression
risk weighted by predicted acceptance, plus a squared shortfall penalty below
80% mean acceptance. This follows the general joint prediction/rejection idea
of [SelectiveNet](https://proceedings.mlr.press/v97/geifman19a.html), but it is a
small disclosed variant, not a reproduction of that paper. Ordinary regression
and correctness supervision remain active so the model cannot freely reject
every difficult training example.

## Separation of development and test information

The generator produces 180 synthetic subjects and six windows per subject.
Every window lasts 20 seconds at 50 Hz. Subjects split into 108 training, 36
validation and 36 test subjects. Only four known scenario families enter
training or validation. Validation divides further into 18 tuning subjects for
epoch selection and 18 calibration subjects for temperature, threshold and
interval fitting. Test subjects contribute 144 ID cases in the known families
and 72 OOD cases in the two entirely held-out families, irregular pulse and
weak perfusion. Training/validation cases from these OOD families are discarded.

The OOD experiment changes equation families within a single simulator; it
does not establish generalization to a new hospital, device, patient population
or real disease. ID and OOD test observations share test subjects but use
different windows/families. Subject-level bootstrapping respects this grouping.

Each test case has five paired clean/stress views and four explicit single
modality-drop views. Stresses preserve the physiological target and case ID.
The random missingness used in adaptation removes ECG, PPG or respiration.
Dropping motion is therefore a deliberately unseen missingness type, which
tests dependence on the artifact side-channel rather than a trained condition.

Three fixed training seeds—17, 29 and 43—are used for both architectures.
Generation, split, stress, validation split and bootstrap seeds are separately
recorded. CPU deterministic kernels with two threads are used. Reproducibility
is expected on the recorded software/hardware; identical results across other
platforms or library versions are not guaranteed.

## Calibration, uncertainty and statistical reporting

The binary correctness and corruption heads each receive one scalar
temperature, selected by Bernoulli NLL from a deterministic grid containing
T=1 on calibration subjects only. This adapts the temperature-scaling approach
of [Guo et al.](https://proceedings.mlr.press/v70/guo17a.html). Calibration-set
NLL cannot worsen relative to T=1 on that grid. Test and OOD NLL can worsen and
are reported. Temperature scaling preserves logit ordering and therefore
cannot improve a model's risk/coverage ranking on its own.

The deployment rejection threshold is the calibration score's 20th percentile,
targeting 80% **aggregate calibration coverage** across the five-condition
mixture. It is frozen before test evaluation. Coverage on an individual test
condition, a held-out family or an unseen modality drop is allowed to differ.
Risk/coverage curves are diagnostic ranking evaluations; their test labels
are not used to choose that threshold. Ranking uses logits to avoid numerical
sigmoid saturation and averages risk under random ordering within exact ties.

Regression bands use per-target **subject-max split conformal** scores. Each
calibration subject contributes its maximum residual over all 20 known-family
and stress views. At target coverage c, the order statistic is
`ceil((m+1)c)` among m calibration subjects. The interval is prediction ± that
score. With m=18, the 90% score is the maximum calibration-subject residual;
these bands are intentionally conservative. A 95% finite score is unavailable
with this calibration size and is returned as null/unbounded if requested.
The exchangeability argument targets the entire fixed collection of a future
ID subject, per target separately. It does not guarantee joint HR/RR coverage
or coverage under OOD families/unseen missingness. Those receive empirical
case and subject coverage alongside width. See the split-conformal framework
in [Angelopoulos and Bates](https://arxiv.org/abs/2107.07511).

The release reports MAE/RMSE in physical units; benchmark failure rate;
available-sensor corruption AUROC, average precision and prevalence; simulator
control-severity MAE; NLL/Brier/10-bin ECE for binary heads; achieved rejection
coverage and accepted-case error; AURC; and interval width plus empirical
coverage. Undefined single-class AUROC/AP is null rather than fabricated.
Control severity is an artificial transform parameter, not disease severity.

For paired comparisons, each model seed's treatment/control absolute-error
difference is averaged within subject, then averaged across the three fixed
seeds. The 2,000-draw bootstrap resamples whole subjects. Reported percentile
95% intervals are conditional on the simulator and fixed seeds; seed standard
deviations are reported separately. Intervals are pointwise and are not
adjusted for the many conditions or outcomes. They should not be treated as
confirmatory hypothesis tests.

## Reproduce and audit

Install the package with its neural dependencies, then run from the repository:

```bash
python -m physiobench.experiments --config configs/experiment-study.json --output results/reproduction
python -m pytest tests/test_training.py -q
```

The smoke configuration verifies wiring only and must not be used as research
evidence. The full study produces 24 fitted model checkpoints and 432 per-run
test evaluations. `configuration.json` stores the exact run arguments;
`training_history.json` stores every epoch and chosen checkpoint;
`per_run_metrics.json` stores each seed/stage/condition; `aggregate_metrics.json`
stores seed means and standard deviations; `paired_subject_bootstrap.json`
stores paired effect sizes and intervals; and `predictions/` preserves every
test prediction with case ID, subject ID and target. `manifest.json` records
hardware, software, data/source hashes and every artifact checksum. Its final
`status: completed` is written only after every analysis is finished.

Checkpoints use independent synthetic observations only. No UCLA source,
patient observations or protected weights enter this training pipeline. A
separate frozen external dataset evaluation, if present, is exploratory domain
transfer and must be interpreted under its own published protocol.

## Measured findings

The numerical findings below are generated from the completed final study.
See `results/neural-study/aggregate_metrics.json` and the paired bootstrap
file for the complete record, including negative outcomes.

The full final study took 68.03 seconds on the recorded CPU configuration.
The CNN and Transformer have 50,699 and 64,763 parameters respectively. Across
three seeds, base CNN clean ID HR MAE is 1.918 bpm and RR MAE is 0.356
breaths/min. Base CNN clean OOD HR MAE rises to 4.078 bpm, while the analogous
Transformer values are 2.239 and 5.310 bpm. The family shift matters even
before observation corruption.

Relative to the matched continued-training control, robustness adaptation
reduces OOD contradiction HR MAE from 16.568 to 11.604 bpm for the CNN and
from 16.350 to 10.133 bpm for the Transformer. Paired subject-bootstrap deltas
are −4.964 bpm [−6.391, −3.414] and −6.217 bpm [−7.079, −5.300]. These are
pointwise exploratory intervals. The corruption always targets PPG in this
generator: reduced error could reflect learning that sensor-specific prior,
not general reasoning about arbitrary contradictory sensors. Contradiction
failure rates remain 84.3% and 77.8% under the declared two-rate correctness
event, so the apparent error improvement does not establish acceptable
reliability.

Adaptation is harmful on some conditions. CNN clean OOD HR MAE increases
from 4.256 to 5.926 bpm relative to continued training, a paired delta of
+1.670 bpm [0.441, 2.911]. Transformer random missing-modality OOD HR MAE
improves from 8.985 to 6.446 bpm, but adding the selective objective worsens
it to 6.827 bpm. Selective adaptation also worsens Transformer OOD
contradiction HR MAE from 10.133 to 10.578 bpm, a paired delta of +0.445 bpm
[0.287, 0.592]. Some CNN conditions improve under selection; no uniform
advantage is supported. All outcomes are retained instead of selecting an
architecture or objective after observing the test.

Dropping the motion stream—an untrained missingness type—causes large errors.
The base CNN clean ID HR MAE rises from 1.918 to 12.553 bpm. This is consistent
with brittle dependence on the motion/amplitude side-channel, although a
mechanistic ablation is needed to distinguish shortcut learning from generic
missingness shift. The frozen selective CNN accepts only 1.4% of those ID
motion-drop cases despite targeting 80% calibration-mixture acceptance.
Reporting accepted-case error without that achieved coverage would be
misleading.

Calibration has mixed transfer behavior. Averaging equally over all nine OOD
conditions and three seeds, base CNN correctness NLL improves from 1.677 to
0.801 after temperature scaling. Robust-adaptation CNN corruption NLL instead
worsens from 0.123 to 0.241 on the same nine-condition summary. The scalar was
fit only to the five-condition calibration mixture; the test summary includes
four additional modality-drop types. This is a negative transfer result,
not a contradiction of calibration-set optimization. Temperature scaling
cannot repair an incorrectly ranked rejector.

The 90% subject-max interval is broad: robust CNN mean full width is 55.40 bpm
for HR and 16.57 breaths/min for RR, even on clean OOD cases. Their empirical
coverage is 100% on those cases, which should be read alongside these widths
and the lack of an OOD exchangeability guarantee. This pilot's small
calibration-subject count and challenging full stress collection limit
useful precision; high coverage alone is not evidence of useful uncertainty.

The separate frozen BIDMC experiment evaluates monitor/annotator references
rather than this simulator's latent window-mean targets, introducing target
mismatch as well as observation/domain shift. Its preselected base CNN HR MAE
is 12.393 bpm, versus 7.130 bpm for signal processing. Respiratory annotation
1 gives 1.495 versus 1.007 breaths/min and annotation 2 gives 1.544 versus
1.017. No real-data model was fitted and no winning checkpoint was selected.
Those results weaken any claim of immediate real-signal transportability.
