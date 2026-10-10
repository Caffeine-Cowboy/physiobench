"""Observable-only spectral features and transparent signal-processing baseline."""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks, welch
from scipy.stats import kurtosis, skew
from .data import PhysioDataset


def spectral_rate(signal: np.ndarray, fs: float, low: float, high: float) -> tuple[float, float]:
    """Welch-band maximum with log-parabolic sub-bin interpolation."""
    if np.std(signal) < 1e-7:
        return 0.0, 0.0
    frequencies, power = welch(signal, fs=fs, nperseg=len(signal), nfft=4*len(signal))
    select = np.flatnonzero((frequencies >= low) & (frequencies <= high))
    if not len(select):
        return 0.0, 0.0
    index = int(select[np.argmax(power[select])])
    frequency = frequencies[index]
    if 0 < index < len(power)-1:
        a, b, c = np.log(np.maximum(power[index-1:index+2], 1e-15))
        denominator = a - 2*b + c
        if abs(denominator) > 1e-12:
            frequency += np.clip(0.5*(a-c)/denominator, -1, 1) * (frequencies[1]-frequencies[0])
    # Integrated local concentration is less sensitive to zero-padding than peak/total.
    local = (frequencies >= frequency-0.10) & (frequencies <= frequency+0.10)
    concentration = float(power[local].sum()/max(float(power[select].sum()), 1e-12))
    return float(frequency*60), min(concentration, 1.0)


def peak_rate(signal: np.ndarray, fs: float) -> tuple[float, float]:
    if np.std(signal) < 1e-7:
        return 0.0, 1.0
    centered = signal - np.median(signal)
    peaks, _ = find_peaks(centered, distance=int(fs*0.32),
                          prominence=max(np.std(centered)*0.9, 0.05))
    if len(peaks) < 3:
        return 0.0, 1.0
    intervals = np.diff(peaks)/fs
    median = float(np.median(intervals))
    # Trim long missed-peak intervals without using any ground-truth information.
    usable = intervals[(intervals > 0.32) & (intervals < 1.8)]
    if len(usable) < 2:
        return 0.0, 1.0
    rate = 60/float(usable.mean())
    cv = float(usable.std()/max(float(usable.mean()), 1e-6))
    return rate, min(cv, 1.0)


def waveform_features(dataset: PhysioDataset) -> np.ndarray:
    """Extract 70 features exclusively from observations and observed masks.

    This function intentionally never accesses targets, severity, corruption,
    scenario names, subject IDs, or latent trajectories.
    """
    all_features = []
    for signals, observed in zip(dataset.x, dataset.observed):
        row = []
        rates = []
        for channel, signal in enumerate(signals):
            sd = float(signal.std())
            normalized = (signal - signal.mean())/max(sd, 1e-6)
            hr, hr_concentration = spectral_rate(signal, dataset.fs, 0.6, 3.0)
            rr, rr_concentration = spectral_rate(signal, dataset.fs, 0.10, 0.8)
            peak_hr, interval_cv = peak_rate(signal, dataset.fs)
            row.extend([
                float(signal.mean()), sd, float(np.quantile(signal, .95)-np.quantile(signal,.05)),
                float(skew(normalized)) if sd > 1e-7 else 0.,
                float(kurtosis(normalized)) if sd > 1e-7 else 0.,
                float(np.abs(np.diff(signal)).mean()),
                hr, hr_concentration, rr, rr_concentration, peak_hr, interval_cv,
                float(np.corrcoef(normalized[:-1], normalized[1:])[0,1]) if sd > 1e-7 else 0.,
                float(observed[channel]),
                float(np.sqrt(np.mean(signal**2))),
                float(np.quantile(np.abs(normalized), .99)),
            ])
            rates.append(hr)
        for left, right in ((0,1),(0,3),(1,3),(2,3)):
            a, b = signals[left], signals[right]
            correlation = float(np.corrcoef(a,b)[0,1]) if a.std()>1e-7 and b.std()>1e-7 else 0.
            row.append(correlation)
        row.extend([abs(rates[0]-rates[1]), float(observed[:3].sum())])
        all_features.append(row)
    return np.nan_to_num(np.asarray(all_features, np.float32), nan=0., posinf=0., neginf=0.)


class SignalBaseline:
    """ECG peak HR + respiration spectral RR; fallback to PPG if ECG absent.

    Constant uncertainty is fit on development subjects, never test subjects.
    Corruption probabilities are transparent heuristics rather than a diagnostic.
    """
    def __init__(self) -> None:
        self.bias = np.zeros(2, np.float32)
        self.std = np.asarray([10., 4.], np.float32)

    def _raw(self, dataset: PhysioDataset) -> tuple[np.ndarray, np.ndarray]:
        means, corruption = [], []
        for signals, observed in zip(dataset.x, dataset.observed):
            ecg_hr, cv = peak_rate(signals[0], dataset.fs)
            ppg_hr, ppg_quality = spectral_rate(signals[1], dataset.fs, .6, 3.)
            rr, resp_quality = spectral_rate(signals[2], dataset.fs, .10, .8)
            if not observed[0] or ecg_hr == 0:
                ecg_hr = ppg_hr if observed[1] and ppg_hr > 0 else 75.
            if not observed[2] or rr == 0:
                # Explicit population fallback makes missing respiration difficult.
                rr = 18.
            motion_sd = float(signals[3].std())
            contradiction = min(abs(ecg_hr-ppg_hr)/30., 1.) if observed[1] else 1.
            quality = np.asarray([
                max(cv, min(motion_sd, 1.)),
                max(contradiction, min(motion_sd*1.2, 1.)),
                1-resp_quality,
                0.02,
            ])
            quality[~observed] = 1.
            means.append([ecg_hr, rr])
            corruption.append(np.clip(quality, .01, .99))
        return np.asarray(means, np.float32), np.asarray(corruption, np.float32)

    def fit(self, dataset: PhysioDataset) -> "SignalBaseline":
        raw, _ = self._raw(dataset)
        residual = dataset.y - raw
        self.bias = residual.mean(axis=0)
        self.std = np.maximum((residual-self.bias).std(axis=0), .5)
        return self

    def predict(self, dataset: PhysioDataset) -> dict[str, np.ndarray]:
        raw, corruption = self._raw(dataset)
        return {"mean": raw + self.bias, "std": np.broadcast_to(self.std, raw.shape).copy(),
                "corruption_prob": corruption, "severity_pred": corruption.copy()}
