# Exploratory external signal validation

The dataset, windows and transfer question were fixed before downloading BIDMC
recordings on 10 October 2026. WFDB punctuation and annotation-format handling
were checked against the acquired file schema before performance evaluation.
The complete execution rules, exact 24-model identities, model hashes and source
hash were frozen before computing any external outcome metrics; the adjacent
machine-readable protocol receipt records this freeze. No choices were revised
after observing the external results.

This experiment tests transfer of signal estimators from stylized synthetic physiology to public ICU waveforms. It is a domain-shift sanity check, not clinical validation and not a confirmatory test of adaptation efficacy. No model selection, normalization fitting, calibration or fine-tuning may use these records.

## Inclusion and observations

Use all 53 records in BIDMC PPG and Respiration Dataset v1.0.0. Use 20-second windows starting at 60, 180 and 300 seconds in each record. Channels are ECG lead II, PLETH and RESP; normalize the trailing commas in WFDB channel names and require exactly one of each. The motion channel is zero with its observed mask false; it must not be represented as a measured stillness signal. Resample 125 Hz signals to 50 Hz with polyphase anti-aliasing. The common inference cohort requires complete finite waveforms and at least one finite monitor HR sample in the inclusive range 30–200 beats/minute. Exclude windows only under these fixed rules and publish counts and reasons. Missing RR annotations do not remove a signal window from prediction; they remove it only from that annotator's RR metric. Consequently RR metrics share the common HR-eligible signal cohort, with additional outcome-specific reference eligibility.

## References and analysis

Heart-rate reference is the median of the monitor samples passing the fixed validity rule during the half-open window. A single available valid sample is sufficient under this protocol, so partial monitor coverage limits interpretation. It is a device-derived reference and does not independently establish diagnostic truth. Respiration reference is the rate derived from manual breath annotations using the median inter-breath interval within the half-open window. Require at least two annotated breaths and use the annotation sampling rate (125 Hz for this source). Evaluate `ann1` and `ann2` separately; do not silently pool duplicate annotations. Duplicate sample positions within an annotator are deduplicated. An unavailable RR reference is never treated as an observed rate: positive placeholder targets needed by the dataset/model interface are excluded from reference scoring, which uses the separately retained reference arrays.

Report equal-recording MAE in beats/minute and breaths/minute: first average window errors within each recording, then average these recording means. Bootstrap whole recording means with 2,000 draws, seed 20261014, and percentile 95% intervals. Report eligible recording/window counts for each reference. If fewer than two recording clusters are eligible, return undefined intervals and a stated reason, rather than inventing a confidence interval or crashing. Source recordings are the clustering unit; independence at the true-patient level cannot be established from the released record inventory. Intervals are pointwise and not multiplicity adjusted across 25 estimators and three references.

Evaluate the fixed signal-processing baseline and every combination of two architectures (CNN, transformer), four stages (base, continued-base, robust adaptation, selective adaptation), and the PhysioBench study seeds 17, 29, 43. Verify the exact set of 24 checkpoint identities, their 50 Hz/1,000-sample metadata, and content hashes before reading external references. Freeze source/protocol/model hashes beside the requested report before scoring. Existing reports or protocol files cannot be overwritten. All models are scored; external outcomes select neither a winner for deployment nor an analysis subset. The signal-processing baseline is fit only on the fixed synthetic training split. No model normalization, calibration or weights may be fit on BIDMC.

Do not infer a corruption label for natural signals or report corruption AUROC on unannotated artifacts. This initial external analysis reports unconditional MAE only; no abstention or calibration benefit is claimed. Any later coverage analysis must accompany reference error and use a development-fixed policy.

Raw records, source subject IDs, demographic header comments and per-record patient-derived predictions stay in a local acquisition directory and are excluded from the public release. Publish code, acquisition hashes and aggregate metrics. Dataset provenance: https://physionet.org/content/bidmc/1.0.0/ . Dataset DOI: 10.13026/C2208R. Attribute the dataset under the Open Data Commons Attribution License v1.0. This project's independently written code uses its own license.
