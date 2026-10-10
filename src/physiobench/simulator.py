"""Stylized independent ECG/PPG/respiration/motion simulator.

The equations encode a shared pulse phase, delayed peripheral pulse, and a
respiratory oscillator. This is a *controlled algorithmic simulator*, not a
biophysical digital twin. All parameters were authored for this repository.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.ndimage import gaussian_filter1d

MODALITIES = ("ecg", "ppg", "respiration", "motion")
KNOWN_SCENARIOS = ("rest", "exertion", "recovery", "respiratory_shift")
OOD_SCENARIOS = ("irregular_pulse", "weak_perfusion")
SCENARIOS = KNOWN_SCENARIOS + OOD_SCENARIOS


@dataclass(frozen=True)
class SubjectParameters:
    resting_hr: float
    resting_rr: float
    pulse_transit_s: float
    ecg_gain: float
    ppg_gain: float
    resp_gain: float
    qrs_width: float
    noise: float


def sample_subject(rng: np.random.Generator) -> SubjectParameters:
    return SubjectParameters(
        resting_hr=float(rng.uniform(52, 86)),
        resting_rr=float(rng.uniform(10, 18)),
        pulse_transit_s=float(rng.uniform(0.12, 0.32)),
        ecg_gain=float(rng.uniform(0.75, 1.35)),
        ppg_gain=float(rng.uniform(0.6, 1.5)),
        resp_gain=float(rng.uniform(0.6, 1.5)),
        qrs_width=float(rng.uniform(0.017, 0.032)),
        noise=float(rng.uniform(0.015, 0.045)),
    )


def _wrapped_gaussian(phase: np.ndarray, center: float, width: float) -> np.ndarray:
    distance = (phase - center + 0.5) % 1.0 - 0.5
    return np.exp(-0.5 * (distance / width) ** 2)


def ecg_from_phase(phase: np.ndarray, width: float) -> np.ndarray:
    # Independent P/Q/R/S/T template, with narrow depolarization and wider T wave.
    return (
        0.10 * _wrapped_gaussian(phase, 0.78, 0.035)
        - 0.16 * _wrapped_gaussian(phase, 0.97, width)
        + 1.15 * _wrapped_gaussian(phase, 0.0, width * 0.72)
        - 0.24 * _wrapped_gaussian(phase, 0.035, width)
        + 0.30 * _wrapped_gaussian(phase, 0.30, 0.065)
    )


def ppg_from_phase(phase: np.ndarray) -> np.ndarray:
    # Smooth systolic pulse plus small delayed reflected component.
    return _wrapped_gaussian(phase, 0.16, 0.080) + 0.26 * _wrapped_gaussian(phase, 0.39, 0.10)


def simulate_window(
    subject: SubjectParameters,
    scenario: str,
    rng: np.random.Generator,
    *,
    fs: float = 50.0,
    duration_s: float = 20.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")
    if fs < 25 or duration_s < 10:
        raise ValueError("Use fs >= 25 Hz and duration >= 10 seconds")
    n = int(round(fs * duration_s))
    t = np.arange(n) / fs
    u = t / duration_s
    hr = np.full(n, subject.resting_hr + rng.uniform(-4, 4))
    rr = np.full(n, subject.resting_rr + rng.uniform(-1.5, 1.5))
    if scenario == "exertion":
        drive = 0.5 - 0.5 * np.cos(np.pi * u)
        hr += rng.uniform(25, 60) * drive
        rr += rng.uniform(8, 17) * drive
    elif scenario == "recovery":
        drive = np.exp(-u * rng.uniform(1.2, 2.5))
        hr += rng.uniform(30, 65) * drive
        rr += rng.uniform(8, 18) * drive
    elif scenario == "respiratory_shift":
        rr += rng.uniform(9, 18) / (1 + np.exp(-(u - 0.45) * 12))
        hr += rng.uniform(1, 5) * np.sin(np.pi * u)
    elif scenario == "irregular_pulse":
        # Frequency modulation is intentionally not an arrhythmia diagnosis.
        smooth_noise = gaussian_filter1d(rng.normal(size=n), sigma=fs * 0.20)
        smooth_noise /= max(float(smooth_noise.std()), 1e-8)
        hr += rng.uniform(12, 24) * smooth_noise
        rr += rng.uniform(0, 3)
    elif scenario == "weak_perfusion":
        hr += rng.uniform(8, 25)
        rr += rng.uniform(2, 8)
    respiratory_phase = rng.uniform(0, 1) + np.cumsum(rr / (60 * fs))
    respiratory_oscillator = np.sin(2 * np.pi * respiratory_phase)
    # Respiratory sinus modulation introduces modest within-window pulse dynamics.
    hr += rng.uniform(1, 3) * respiratory_oscillator
    hr = np.clip(hr, 35, 180)
    rr = np.clip(rr, 6, 45)
    pulse_phase = rng.uniform(0, 1) + np.cumsum(hr / (60 * fs))
    delayed_phase = np.interp(
        t - subject.pulse_transit_s, t, pulse_phase,
        left=pulse_phase[0] - subject.pulse_transit_s * hr[0] / 60,
    )
    baseline = 0.04 * np.sin(2 * np.pi * 0.18 * t + rng.uniform(0, 2*np.pi))
    ecg = subject.ecg_gain * ecg_from_phase(pulse_phase, subject.qrs_width) + baseline
    ppg_amplitude = subject.ppg_gain * (1 + 0.10 * respiratory_oscillator)
    if scenario == "weak_perfusion":
        ppg_amplitude *= rng.uniform(0.10, 0.25)
    ppg = ppg_amplitude * ppg_from_phase(delayed_phase) + 0.07 * respiratory_oscillator
    respiration = subject.resp_gain * (
        respiratory_oscillator + 0.10 * np.sin(4 * np.pi * respiratory_phase + 0.2)
    )
    motion = 0.03 * rng.normal(size=n)
    if scenario == "exertion":
        motion += 0.06 * np.sin(2*np.pi*rng.uniform(0.7, 1.2)*t)
    x = np.stack([ecg, ppg, respiration, motion])
    x += rng.normal(size=x.shape) * subject.noise
    y = np.asarray([hr.mean(), rr.mean()], dtype=np.float32)
    latent = {"hr_bpm": hr.astype(np.float32), "rr_bpm": rr.astype(np.float32)}
    return x.astype(np.float32), y, latent
