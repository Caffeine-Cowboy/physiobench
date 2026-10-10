# PhysioBench synthetic data card

## Identity and provenance

Version 0.1.0 is independently authored, fully synthetic research data. No UCLA
software, records, protected data, confidential methods, or trained weights are
used. A "subject" is an independent draw of simulator parameters, not a person.
Signals use arbitrary amplitude units. ECG, PPG, respiration, and one stylized
motion channel are sampled synchronously at 50 Hz for 20 seconds.

## Intended use

Controlled research on multimodal rate estimation, observation corruption,
uncertainty calibration, selective prediction, distribution shift, and explicit
failure analysis. The benchmark is not intended to predict disease, triage
patients, claim diagnostic validity, or estimate clinical severity. Simulator
severity means the normalized transformation amplitude.

## Generation

Each subject draws resting heart/respiration rates, peripheral pulse transit
delay, signal gains, ECG template width, and additive noise. The independent
equations integrate latent heart and respiratory frequency trajectories. An ECG
Gaussian P/Q/R/S/T template follows the shared cardiac phase; a delayed smooth
peripheral pulse includes a reflected component and respiratory amplitude
modulation. Respiration follows a primary oscillator and weak harmonic. Heart
rate has modest respiratory sinus modulation. Additive noise and baseline wander
are present. These are stylized oscillators; they do not model cardiovascular
fluid dynamics, electrophysiology, tissue optics, a sensor's full transfer
function, or motion kinematics.

The six scenario families are rest, increasing exertion drive, recovery,
respiratory rate shift, irregular pulse frequency modulation, and weak PPG
perfusion amplitude. The last two are held-out simulator families. "Irregular"
does not name or simulate a specific arrhythmia diagnosis.

HR/RR targets are duration averages of the latent trajectories. They are kept in
`labels.npz`, together with corruption/control labels and latent trajectories.
`observations.npz` contains signals, masks, and grouping metadata. The feature
extractor accepts a dataset but reads only waveforms, masks, and sample rate;
tests mutate all labels and metadata to verify features remain identical.

## Paired perturbations

Every stress transform retains the underlying case ID, subject, HR/RR, and latent
trajectories. Randomness is keyed by case ID, condition, and seed, so subsetting
or reordering cannot change a stress realization.

- Missing modality: one ECG, PPG, or respiration channel is zero-filled and its
  observed mask is false.
- Motion artifact: a mixed periodic/noise artifact appears in motion and couples
  into ECG and PPG. The motion sensor is not labeled corrupt: it observes motion.
- Temporal offset: PPG is delayed by 0.8–2.4 seconds with circular continuation.
  Boundary seams are a limitation of finite-window offsets; time-lag detection
  performance must not be interpreted as validated sensor synchronization.
- Contradiction: PPG follows a decoy pulse rate while the electrical heart-rate
  truth is unchanged. The construction assumes ECG represents the target truth;
  real discordance can have physiological causes and requires different labels.

## Coverage and blind spots

No demographic attributes, diseases, medications, devices, skin properties,
hospital setting, posture, clinical outcomes, or cross-patient sensor differences
are represented. The simulator has a small set of hand-defined families and may
reward its own waveform assumptions. Subject variation is narrower and more
regular than real physiology. Synthetic OOD is only a simulator-family shift,
not evidence of generalization to patients or medical practice. External transfer
must be reported separately and should not be used to tune this simulator.

## Reproducibility and access

The simulator is deterministic for a pinned package environment and seed.
Manifests list dimensions and SHA-256 digests; loading verifies both signal and
label files. Generated data can be regenerated from released code. No proprietary
datasets are required. Exact dependency versions for the executed experiment
belong in its runtime manifest because low-level numeric behavior can vary.
