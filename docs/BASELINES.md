# Baseline comparator protocol

`python -m physiobench.baseline_experiment --config configs/experiment-study.json`
regenerates the exact same subjects and paired test conditions as the neural
study. Results are a control experiment, not a test-selected competing system.

The signal comparator estimates ECG HR from detected peak intervals and RR from
the respiration spectral maximum, with a PPG fallback for absent ECG and a fixed
18-bpm RR fallback for absent respiration. Training-subject mean residual bias
is the only fitted rate parameter. Its corruption score combines observable
motion amplitude, peak interval variability, and ECG/PPG disagreement. It cannot
reliably detect every temporal offset. Its severity output is explicitly the
same heuristic score, not a learned clinical severity model.

The feature comparator receives 70 waveform/mask features: each channel's
amplitude/morphology summaries, HR/RR spectral peaks and concentrations, peak
interval statistics, lag-one correlation, availability, selected cross-channel
correlations, and ECG/PPG rate disagreement. No label, latent state, subject ID,
case ID, or scenario is a feature. Tests mutate all such data and require the
feature matrix to remain exactly unchanged.

The fixed random-forest configuration is 128 trees, maximum depth 16, minimum
leaf size 2, and two CPU workers. Seeds are 17/29/43. Train-subject views comprise
clean, motion, missing, offset, and contradiction. The multioutput regressor uses
fixed target scales HR/40 and RR/10 jointly with the four simulator severity
controls; a separate multioutput classifier estimates corruption. The scales
prevent heart-rate units from dominating every impurity split. No tree setting
is selected using test data. The model is intentionally a strong low-cost
comparator, showing whether waveform neural training adds measurable value.

Prediction bands use the same 18-subject calibration fold as the neural study:
per-target subject maximum absolute residual over the fixed scenario/stress
collection, followed by the finite-sample conformal order statistic. They are
fixed-width marginal bands for each rate, conservative for the matching ID
collection. OOD families and the explicit modality ablations receive empirical
coverage reporting; the exchangeability guarantee does not apply there.

All-modalities corruption metrics include the known missingness state; learned
available-sensor corruption metrics separately remove absent channels and match
the neural evaluation convention. AUROC is null for a single-class modality and
AP is accompanied by prevalence. Simulator-control severity MAE is reported for
all channels and separately for observed channels. Neither is disease severity.

The report contains per-seed errors and equal-subject 95% bootstrap intervals.
Paired random-forest-minus-signal differences average fixed-seed errors within
subject before resampling subjects. They are pointwise intervals conditional on
the simulator and fixed seeds. They do not establish medical benefit or account
for simulator misspecification. Checkpoints are not necessary to rerun these
small deterministic fits; all training settings, inputs' checksum, exact
dependency versions, predictions, and artifact checksums are released.
