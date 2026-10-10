"""Validation-only calibration and error-aware selective prediction utilities."""
from __future__ import annotations

import numpy as np


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))


def fit_temperature(logits: np.ndarray, labels: np.ndarray,
                    mask: np.ndarray | None = None) -> dict[str, float]:
    """Select a scalar temperature by Bernoulli NLL on calibration subjects.

    A finite preregistered grid makes optimization deterministic and includes
    T=1, so calibration-set NLL cannot become worse by the selected operation.
    This guarantee does not extend to unseen subjects or OOD scenarios.
    """
    logits, labels = np.asarray(logits, dtype=np.float64), np.asarray(labels, dtype=np.float64)
    if mask is not None:
        logits, labels = logits[np.asarray(mask, dtype=bool)], labels[np.asarray(mask, dtype=bool)]
    if logits.size == 0:
        raise ValueError("Calibration needs observed labels")
    if logits.shape != labels.shape:
        raise ValueError("Logits and labels must have matching shapes")
    candidates = np.unique(np.r_[np.geomspace(0.2, 5.0, 81), 1.0])
    nll = np.array([np.mean(np.logaddexp(0.0, logits / t) - labels * logits / t) for t in candidates])
    index = int(np.argmin(nll))
    before = float(np.mean(np.logaddexp(0.0, logits) - labels * logits))
    return {"temperature": float(candidates[index]), "calibration_nll_before": before,
            "calibration_nll_after": float(nll[index]), "calibration_count": int(logits.size)}


def binary_metrics(logits: np.ndarray, labels: np.ndarray, temperature: float = 1.0,
                   mask: np.ndarray | None = None, bins: int = 10) -> dict[str, float | None]:
    labels = np.asarray(labels, dtype=float)
    probabilities = sigmoid(np.asarray(logits) / temperature)
    if mask is not None:
        selected = np.asarray(mask, dtype=bool)
        labels, probabilities = labels[selected], probabilities[selected]
    labels, probabilities = labels.ravel(), probabilities.ravel()
    if labels.size == 0:
        return {"nll": float("nan"), "brier": float("nan"), "ece": float("nan")}
    clipped = np.clip(probabilities, 1e-7, 1 - 1e-7)
    nll = -np.mean(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped))
    ece = 0.0
    for lower, upper in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        selected = (probabilities >= lower) & (probabilities < upper if upper < 1 else probabilities <= upper)
        if np.any(selected):
            ece += np.mean(selected) * abs(np.mean(probabilities[selected]) - np.mean(labels[selected]))
    positive = int(labels.sum())
    negative = int(labels.size - positive)
    auc, average_precision = None, None
    if positive and negative:
        # Average ranks explicitly handle ties; AP integrates the stepwise
        # precision/recall curve at unique thresholds rather than breaking ties.
        order = np.argsort(probabilities, kind="stable")
        sorted_scores = probabilities[order]
        ranks = np.empty(len(order), dtype=float)
        boundaries = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1, len(order)]
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            ranks[order[start:end]] = 0.5 * (start + 1 + end)
        auc = float((ranks[labels.astype(bool)].sum() - positive * (positive + 1) / 2) / (positive * negative))
        descending = np.argsort(-probabilities, kind="stable")
        sorted_labels, sorted_scores = labels[descending], probabilities[descending]
        ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(sorted_scores) - 1]
        true_positives = np.cumsum(sorted_labels)[ends]
        recall = true_positives / positive
        precision = true_positives / (ends + 1)
        average_precision = float(np.sum(np.diff(np.r_[0.0, recall]) * precision))
    return {"nll": float(nll), "brier": float(np.mean((probabilities - labels) ** 2)), "ece": float(ece),
            "auroc": auc, "average_precision": average_precision, "prevalence": float(labels.mean()),
            "positive_count": positive, "negative_count": negative}


def fit_subject_conformal(predictions: np.ndarray, targets: np.ndarray, subject_ids: np.ndarray,
                          levels: tuple[float, ...] = (0.5, 0.8, 0.9)) -> dict[str, object]:
    """Subject-max split-conformal bands per rate for clustered windows.

    Each subject contributes its maximum residual over the fixed calibration
    scenario/stress collection. The finite-sample order statistic therefore
    targets coverage of an entire new exchangeable subject's collection for
    each rate separately. OOD families and unseen ablations violate this
    exchangeability premise and receive empirical coverage only.
    """
    residual = np.abs(np.asarray(predictions) - np.asarray(targets))
    unique = np.unique(subject_ids)
    scores = np.stack([residual[np.asarray(subject_ids) == subject].max(axis=0) for subject in unique])
    intervals = {}
    for level in levels:
        if not 0 < level < 1:
            raise ValueError("Conformal coverage level must be in (0,1)")
        rank = int(np.ceil((len(unique) + 1) * level))
        # No finite guaranteed band if the required order statistic exceeds n.
        width = np.sort(scores, axis=0)[rank - 1].tolist() if rank <= len(unique) else None
        intervals[str(level)] = {"half_width_bpm": width, "order_statistic": rank}
    return {"method": "per-target subject-max split conformal", "n_calibration_subjects": len(unique),
            "levels": intervals, "guarantee_scope": "new exchangeable synthetic ID subject; per target separately"}


def correctness(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    residual = np.abs(np.asarray(predictions) - np.asarray(targets))
    return ((residual[:, 0] <= 5.0) & (residual[:, 1] <= 3.0)).astype(float)


def acceptance_threshold(calibration_logits: np.ndarray, temperature: float = 1.0,
                         target_coverage: float = 0.8) -> float:
    if not 0 < target_coverage <= 1:
        raise ValueError("Coverage must be in (0,1]")
    return float(np.quantile(sigmoid(np.asarray(calibration_logits) / temperature), 1 - target_coverage))


def risk_coverage_curve(logits: np.ndarray, predictions: np.ndarray, targets: np.ndarray,
                        temperature: float = 1.0) -> dict[str, object]:
    """Diagnostic ranking curve; labels are never used to choose a deployment threshold."""
    # Rank logits directly: sigmoid saturation must not create artificial ties.
    quality = np.asarray(logits, dtype=float) / temperature
    order = np.argsort(-quality, kind="stable")
    failure = 1.0 - correctness(predictions, targets)[order]
    count = np.arange(1, len(order) + 1)
    cumulative = np.cumsum(failure)
    sorted_quality = quality[order]
    boundaries = np.r_[0, np.flatnonzero(np.diff(sorted_quality)) + 1, len(order)]
    # Expected risk under uniformly random ordering within an exact score tie.
    # Constant scores consequently cannot benefit from simulator case ordering.
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        previous = cumulative[start - 1] if start else 0.0
        cumulative[start:end] = previous + np.arange(1, end - start + 1) * failure[start:end].mean()
    risk = cumulative / count
    coverages = np.linspace(0.1, 1.0, 10)
    indices = np.maximum(np.ceil(coverages * len(order)).astype(int) - 1, 0)
    return {"coverage": coverages.tolist(), "failure_rate": risk[indices].tolist(),
            "aurc": float(risk.mean()), "tie_convention": "expected uniform ordering within exact logit ties"}
