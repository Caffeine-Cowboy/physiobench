# Corvus and PhysioBench: reproducible evidence for physiological machine learning

Adrian Moreno · Independent research · Version 0.1.0 · 10 October 2026

## Abstract

We release two complementary research artifacts. Corvus packages independently owned ECG engineering with a newly executed three-seed repeatability analysis on an already observed holdout. PhysioBench implements independent synthetic ECG, peripheral pulse, respiration and motion generation, subject-disjoint evaluation, paired stress interventions, signal-processing and observable-feature comparators, and a 24-fit neural training/adaptation study. The work emphasizes falsifiable comparisons, frozen evaluation protocols, reproducibility and negative results. Broader stress exposure reduces some contradiction errors while degrading clean performance; selective adaptation does not consistently improve the matched ordinary adaptation control. Exploratory frozen-model transfer to open ICU recordings tests the synthetic-to-real gap. These releases are research prototypes; they do not establish clinical efficacy, doctoral certification, scientific novelty or frontier language-model training experience.

## Research questions

1. How variable is a supervised ECG classifier across fixed training seeds when subject membership, preprocessing, labels and validation selection are unchanged?
2. Does a multimodal rate estimator benefit from broader stress exposure beyond an equal optimizer-update budget?
3. Does an acceptance-weighted objective improve a model given the same stress examples and updates?
4. Do confidence and prediction intervals remain informative under missing sensors and held-out equation families?
5. How far do synthetic estimators transfer to a public ICU waveform dataset without any external fitting?

## Corvus evidence

The new repeatability run trains a CNN for 30 epochs for seeds 17, 42 and 173. The fixed split contains 574 training, 192 validation and 193 test patients, with approximately balanced PVC/normal machine-report labels. The validation split selects checkpoint and operating threshold. All three fits finish before test scoring; the report records protocol, source, split and checkpoint hashes. Mean test AUROC is 0.959336 with sample seed SD 0.006351 and a paired patient-bootstrap percentile 95% interval of 0.932866–0.982007. Mean balanced accuracy is 0.903422.

This is training-seed repeatability on a previously inspected holdout, not an untouched confirmatory test. Labels are machine-report-derived and the convenience sample does not represent all ECG diagnostic conditions. The historical astronomy-to-ECG transfer pilot achieved AUROC 0.447266, and its failure remains visible. Historical serving/ONNX/latency measurements are separately identified rather than presented as newly reproduced results. Raw records, subject IDs, prepared examples, weights and individual patient predictions are excluded from the Corvus release. Source-only release checks passed 70 tests in a fresh pinned environment and a synthetic smoke run.

## Synthetic multimodal experiment

PhysioBench uses independent stylized signal equations. ECG, PPG, respiration and motion have shared physiological drivers, modality-specific transformations and subject-dependent nuisance parameters. These equations are deliberately simple and do not implement a validated cardiovascular digital twin or ECGSYN. Every synthetic subject contributes six 20-second windows at 50 Hz. The 180 subjects split into 108 training, 36 validation and 36 test subjects. Only four known families enter development; irregular-pulse and weak-perfusion families are excluded from training and validation. Test subjects contribute 144 known-family windows and 72 unseen-family windows. Validation divides into 18 tuning and 18 calibration subjects.

Observations and supervision are physically separated in serialized datasets. Models receive only waveforms and missingness masks. Hidden labels, identities, simulator parameters and family names never enter inference. Counterfactual tests alter all hidden quantities while preserving inputs and verify identical predictions. The contradiction condition always corrupts PPG while retaining the ECG-derived latent HR target. Improvement can therefore reflect learned reliance on ECG rather than general conflict resolution or physiological reasoning. Paired interventions preserve physiological target and case identity: motion-coupled artifact, random missing modality, shifted PPG, contradictory PPG, and four explicit modality-removal controls. Case-hashed randomness makes stress invariant to dataset order or subsetting. Circular temporal shifts create finite-window seams, an explicitly disclosed limitation.

## Training and adaptation

CNN and temporal Transformer estimators share a 0.1–4 Hz observable spectral branch and per-window normalization. This disclosed inductive bias means they are small spectral-augmented estimators, not raw-waveform foundation models. Both estimate heart and respiration rates, available-sensor corruption, synthetic control severity and prediction correctness. Missing sensors are supplied in the observed mask, so missingness is not claimed as learned detection. The normalized Huber regression objective is combined with available-sensor corruption, severity and correctness supervision. Correctness uses fixed benchmark tolerances of HR error ≤5 bpm and RR error ≤3 breaths/min; these are not clinical acceptance thresholds.

For each architecture and seed (17, 29, 43), a base model trains for 24 epochs on clean plus motion observations. Three alternative eight-epoch stages start from that same base checkpoint: continued clean/motion training with repeated observations; broader five-condition supervised adaptation; and the same five-condition adaptation with selective risk and coverage penalty. All three eight-epoch adaptation stages receive equal example counts and optimizer-update budgets. Robust minus continued-base therefore measures the broader stress-data package; selective minus robust isolates the objective. They are alternative stages, not a sequential pipeline. AdamW, gradient clipping, deterministic CPU kernels, fixed source/configuration/data hashes and full per-epoch histories support rerunning. Epoch selection uses tuning subjects only.

## Comparators and uncertainty

The fixed signal-processing estimator uses ECG peak timing, PPG fallback and respiration spectral estimation, with train-only residual bias fitting. A 70-feature random forest uses observable statistics and signal features; three seeds fit on five paired training stress conditions. It is a feature comparator with broader training exposure, rather than a matched neural architecture or compute control. All comparators are evaluated on the same synthetic cases. The RF clean-ID HR error is approximately 1.839 bpm, competitive with the CNN base; therefore the release makes no universal neural superiority claim.

Temperature scaling of binary correctness/corruption logits and an acceptance threshold are fit on separate calibration subjects. Regression intervals use per-target subject-maximum residuals and a finite-sample split-conformal rank, with 18 calibration subjects. Coverage claims depend on exchangeability and apply to a target-specific subject-group event on the calibration construction; they do not guarantee joint HR/RR coverage or real/OOD validity. Width accompanies coverage because large bands can cover without being useful. Risk/coverage curves are diagnostics; test errors never choose an operational threshold. Single-class corruption AUROC is undefined and reported as null; AUPRC must be read with the reported prevalence. Severity denotes simulator intervention magnitude, not clinical disease severity.

Paired bootstrap comparisons resample the 36 held-out subjects and average paired case errors across the three training seeds. The intervals quantify subject uncertainty conditional on those seeds and the simulator, not uncertainty over populations, devices or arbitrary training randomness. The 2,000-draw pointwise percentile intervals are exploratory and unadjusted for multiple comparisons. Source data and synthetic held-out predictions permit independent reanalysis.

## Measured findings

CNN base clean-ID mean HR MAE is 1.918 bpm and RR MAE is 0.356 breaths/min. The clean unseen-family HR error is 4.078 bpm. Broader adaptation improves unseen-family contradiction HR error versus matched continued training: CNN -4.964 bpm (pointwise 95% CI -6.391 to -3.414); Transformer -6.217 bpm (pointwise 95% CI -7.079 to -5.300).

The same intervention worsens CNN clean unseen-family HR error by +1.670 bpm (pointwise 95% CI +0.441 to +2.911). Selective adaptation worsens Transformer unseen-family contradiction error relative to ordinary stress adaptation by +0.445 bpm (pointwise 95% CI +0.287 to +0.592). Contradiction errors remain large, so improvement relative to control does not establish reliable reasoning. Calibration can improve its own calibration objective while worsening error-confidence alignment elsewhere. Full results retain all conditions, seeds, intervals and failure metrics rather than selecting favorable cells.

## Exploratory real-signal validation

The separately frozen protocol uses all 53 records of BIDMC PPG and Respiration Dataset v1.0.0 and 20-second windows starting at 60, 180 and 300 seconds. ECG lead II, PLETH and RESP resample from 125 to 50 Hz with polyphase antialiasing; absent motion is zero-filled with an explicit false mask. Monitor HR uses a valid-range median; manual respiration annotations are kept separate for annotators ann1 and ann2 and require at least two breaths. A frozen signal baseline and every final neural checkpoint are evaluated without training, tuning, selection or recalibration on BIDMC. CNN base seed 17 is displayed alongside the signal comparator, and all 24 models are reported without selecting a deployment candidate.

The aggregate external report provides inclusion/exclusion counts and recording-cluster MAE intervals. Synthetic targets are duration-average latent rates, whereas external references use monitor-HR medians and median inter-breath intervals; different estimands further limit direct comparison. Monitor-derived HR is an imperfect reference; manual breath intervals have annotation and finite-window limitations. Manual breaths were annotated from the same observed respiration signal, so RR is an annotated signal-estimation target rather than an independent clinical outcome. Recording groups cannot be assumed to establish independent patient identity. Natural corruption has no annotation truth, so detector AUROC is not reported. Raw records, demographic header comments, record IDs and individual external predictions stay local and are excluded from distribution. The dataset is open under Open Data Commons Attribution License v1.0; the report attributes its creators and DOI 10.13026/C2208R.


Measured external result: 53 recordings and all 159 planned signal windows are eligible. Manual RR scoring includes 157 windows for ann1 and 158 for ann2. The prespecified CNN base seed 17 has HR MAE **12.393 bpm** (recording-bootstrap 95% CI 10.986–13.705), versus **7.130 bpm** (4.568–10.155) for the signal-processing baseline. Its ann1/ann2 RR errors are **1.495 / 1.544 breaths/min**, versus **1.007 / 1.017** for the signal baseline. The synthetic CNN does not improve these external estimands; no external model ranking selects a deployment candidate.

## Reproducibility and ownership

Each release has a package definition, environment snapshot, tests, acquisition/preparation code where applicable, configuration, provenance and license. Corvus is source-only plus aggregate evidence. PhysioBench additionally releases synthetic checkpoints and predictions. No UCLA-controlled code or protected data were copied into this work. Public data acquisition is explicit and local; license and access restrictions must be reviewed before extending to a different dataset. Local verification receipts distinguish completed checks from unexecuted remote CI. PhysioBench passes 44 tests, verifies 560 published file hashes, and reproduces all 432 neural prediction files plus all 24 model states and study metrics bitwise in a second full run on the same reference environment. A four-fit smoke study also executes successfully. This is same-environment repeatability rather than independent external replication. Seed/hardware/software changes may prevent bitwise equality, so dataset hashes and environment details must accompany replication claims.

## Research progression and acceptance criteria

**Independent replication.** A second environment should install the snapshots, pass scientific invariants, regenerate the simulator and reproduce frozen aggregate results. Reconcile any mismatch before treating the release as replicable beyond the recorded environment.

**Self-supervised multimodal representation learning.** Add masked reconstruction and ECG/PPG contrastive objectives with explicit masking policies and subject-balanced batches. Compare initialization to supervised learning with equal downstream updates and measured pretraining compute. Hold out subjects and signal families before representation development. Publish ablations for spectral branch, motion side-channel, loss weights and mask geometry, including negative results. Improvement must persist in multiple seeds and at least one genuinely separate public waveform distribution, without tuning its final test set.

**Language-model training and post-training.** Create synthetic tool-grounded question/answer records that require numerical signal estimates and cited observable evidence. Validate every target with independent executable checks. Train a small licensed open language model using SFT with held-out subjects, scenario families and question templates. Compare continued SFT to preference optimization on verified correctness preferences at matched compute; record reward failures and response-length effects. Measure numerical error, unsupported evidence, contradiction handling, abstention calibration and performance without tools. This is planned future work, not completed LLM training experience.

**External transport and reference validity.** Freeze an external protocol and sample-size rationale before accessing outcomes. Use independent labels and public licenses; assess subgroup/device and missingness shifts where provenance permits. Calibrate on a separate external development cohort only if adaptation is explicitly the research question. Keep an untouched evaluation cohort. Clinical claims require a substantially different validation program.

## References

- [BIDMC PPG and Respiration Dataset v1.0.0](https://physionet.org/content/bidmc/1.0.0/). Pimentel et al.; DOI 10.13026/C2208R. External dataset provenance and attribution.
- [ECGSYN model paper](https://archive.physionet.org/physiotools/ecgsyn/paper/). McSharry et al. (2003). Related synthetic ECG work; this release does not reproduce that model.
- [How You Split Matters](https://arxiv.org/abs/2309.00350). Rumala (2023). Related evidence about subject-level leakage in longitudinal biomedical modeling.
- [SelectiveNet](https://proceedings.mlr.press/v97/geifman19a.html). Geifman and El-Yaniv (2019). Related joint prediction/rejection formulation; our objective is a disclosed variant.

The report is an editable research artifact, not a peer-reviewed publication. Public evidence is a basis for evaluation and further work rather than a claim of readiness for an unrestricted health application.
