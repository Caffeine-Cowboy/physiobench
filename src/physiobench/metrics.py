"""Regression, confidence, corruption, abstention, and subject-cluster inference."""
from __future__ import annotations

from typing import Callable
import numpy as np
from scipy.stats import norm
from sklearn.metrics import average_precision_score, roc_auc_score
from .data import PhysioDataset


def binary_ece(truth: np.ndarray, probability: np.ndarray, n_bins: int = 10) -> float:
    truth, probability = np.asarray(truth).ravel(), np.asarray(probability).ravel()
    if not len(truth):
        return float("nan")
    edges = np.linspace(0, 1, n_bins+1)
    result = 0.
    for i, (low, high) in enumerate(zip(edges[:-1],edges[1:])):
        select = (probability >= low) & ((probability <= high) if i==n_bins-1 else (probability < high))
        if select.any():
            result += float(select.mean()*abs(truth[select].mean()-probability[select].mean()))
    return result


def selective_risk_curve(y: np.ndarray, mean: np.ndarray, std: np.ndarray,
                         coverages: tuple[float,...] = (.25,.5,.75,1.0)) -> dict[str, float]:
    # Standardize task units using *fixed* protocol tolerances, not test dispersion.
    scale = np.asarray([5., 2.])
    uncertainty = (std/scale).mean(axis=1)
    ordered = np.sort(uncertainty)
    errors = (np.abs(y-mean)/scale).mean(axis=1)
    result = {}
    for coverage in coverages:
        count = max(1,int(len(y)*coverage))
        threshold = ordered[count-1]
        below, tied = uncertainty < threshold, uncertainty == threshold
        # Expected risk for random acceptance within the boundary-score tie.
        # Constant uncertainty therefore has constant risk, rather than inheriting
        # an arbitrary scenario/order preference from argsort.
        total = errors[below].sum() + (count-below.sum()) * errors[tied].mean()
        result[f"coverage_{coverage:.2f}"] = float(total/count)
    return result


def evaluate_predictions(
    dataset: PhysioDataset, mean: np.ndarray | dict[str,np.ndarray],
    std: np.ndarray | None = None, corruption_prob: np.ndarray | None = None,
    severity_pred: np.ndarray | None = None, abstain_threshold: float = 1.0,
) -> dict:
    """Evaluate predictions; passing a dictionary from baseline.predict is supported.

    std represents predictive SD, not a confidence interval on the mean. Gaussian
    NLL/coverage test the stated distribution rather than validating physiology.
    """
    if isinstance(mean, dict):
        prediction = mean
        mean, std = prediction["mean"], prediction.get("std")
        corruption_prob = prediction.get("corruption_prob")
        severity_pred = prediction.get("severity_pred")
    mean = np.asarray(mean)
    if mean.shape != dataset.y.shape or not np.isfinite(mean).all():
        raise ValueError("Mean must be finite [N,2]")
    result: dict = {"n_cases":len(dataset),"n_subjects":len(np.unique(dataset.subject_id))}
    for j, name in enumerate(("hr", "rr")):
        error = mean[:,j]-dataset.y[:,j]
        result[f"{name}_mae"] = float(np.abs(error).mean())
        result[f"{name}_rmse"] = float(np.sqrt(np.mean(error**2)))
        result[f"{name}_bias"] = float(error.mean())
    if std is not None:
        std = np.asarray(std)
        if std.shape != mean.shape or not np.isfinite(std).all() or np.any(std <= 0):
            raise ValueError("std must be finite positive [N,2]")
        z = (dataset.y-mean)/std
        result["gaussian_nll"] = float((.5*z**2 + np.log(std) + .5*np.log(2*np.pi)).mean())
        for j, name in enumerate(("hr", "rr")):
            for coverage in (.50,.80,.90,.95):
                threshold = norm.ppf((1+coverage)/2)
                result[f"{name}_coverage_{coverage:.2f}"] = float((np.abs(z[:,j])<=threshold).mean())
            result[f"{name}_mean_std"] = float(std[:,j].mean())
        result["selective_risk"] = selective_risk_curve(dataset.y, mean, std)
        normalized_uncertainty = (std/np.asarray([5.,2.])).mean(axis=1)
        accept = normalized_uncertainty <= abstain_threshold
        result["abstention_coverage"] = float(accept.mean())
        result["accepted_normalized_mae"] = (float((np.abs(dataset.y[accept]-mean[accept])/np.asarray([5.,2.])).mean())
                                              if accept.any() else None)
    if corruption_prob is not None:
        probability = np.asarray(corruption_prob)
        if probability.shape != dataset.corruption.shape or not np.isfinite(probability).all() or np.any((probability<0)|(probability>1)):
            raise ValueError("Corruption probability must be finite [N,4] in [0,1]")
        scores = {}
        for j, name in enumerate(("ecg","ppg","respiration","motion")):
            truth = dataset.corruption[:,j]
            # Single-class tests have undefined discrimination; never replace by 0/1.
            scores[name] = {
                "auroc": float(roc_auc_score(truth, probability[:,j])) if len(np.unique(truth))==2 else None,
                "auprc": float(average_precision_score(truth, probability[:,j])) if truth.any() else None,
                "brier": float(np.mean((probability[:,j]-truth)**2)),
                "ece": binary_ece(truth,probability[:,j]),
                "prevalence": float(truth.mean()),
            }
        defined = [row["auroc"] for row in scores.values() if row["auroc"] is not None]
        result["corruption"] = scores
        result["corruption_macro_auroc"] = float(np.mean(defined)) if defined else None
        result["corruption_brier"] = float(np.mean((probability-dataset.corruption)**2))
    if severity_pred is not None:
        predicted = np.asarray(severity_pred)
        if predicted.shape != dataset.severity.shape or not np.isfinite(predicted).all():
            raise ValueError("Severity prediction must be finite [N,4]")
        result["severity_mae"] = float(np.abs(predicted-dataset.severity).mean())
        positive = dataset.corruption
        result["severity_mae_corrupted"] = float(np.abs(predicted[positive]-dataset.severity[positive]).mean()) if positive.any() else None
    return result


def cluster_bootstrap_mean(values: np.ndarray, subject_id: np.ndarray,
                           n_bootstrap: int = 2000, seed: int = 20261010) -> dict[str,float]:
    """Equal-subject estimand and percentile bootstrap resampling whole subjects.

    Each subject contributes its within-subject mean. This prevents a subject with
    more windows from silently dominating the effect estimate.
    """
    values, subject_id = np.asarray(values), np.asarray(subject_id)
    if len(values) != len(subject_id) or values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Provide finite scalar values and matching subject IDs")
    if n_bootstrap < 100:
        raise ValueError("Use >=100 bootstrap replicates")
    subjects = np.unique(subject_id)
    if len(subjects) < 2:
        raise ValueError("At least two independent subject clusters required")
    cluster_means = np.asarray([values[subject_id==subject].mean() for subject in subjects])
    rng = np.random.default_rng(seed)
    bootstrap = cluster_means[rng.integers(0,len(subjects),size=(n_bootstrap,len(subjects)))].mean(axis=1)
    low, high = np.quantile(bootstrap,[.025,.975])
    return {"estimate":float(cluster_means.mean()),"ci95_low":float(low),"ci95_high":float(high),
            "n_subjects":int(len(subjects)),"n_bootstrap":int(n_bootstrap)}


def paired_cluster_bootstrap(
    reference: PhysioDataset, candidate: PhysioDataset,
    reference_pred: np.ndarray, candidate_pred: np.ndarray,
    target: int = 0, n_bootstrap: int = 2000, seed: int = 20261010,
) -> dict[str,float]:
    """Candidate minus reference absolute-error change; negative means improvement."""
    if len(set(reference.case_id)) != len(reference) or len(set(candidate.case_id)) != len(candidate):
        raise ValueError("Paired inference needs one row per unique underlying case")
    lookup = {case:i for i,case in enumerate(candidate.case_id)}
    if set(reference.case_id) != set(candidate.case_id):
        raise ValueError("Paired datasets must contain identical case IDs")
    order = np.asarray([lookup[case] for case in reference.case_id])
    if not np.array_equal(reference.subject_id,candidate.subject_id[order]) or not np.array_equal(reference.y,candidate.y[order]):
        raise ValueError("Paired datasets have mismatched subjects or physiological truth")
    reference_error = np.abs(np.asarray(reference_pred)[:,target]-reference.y[:,target])
    candidate_error = np.abs(np.asarray(candidate_pred)[order,target]-reference.y[:,target])
    return cluster_bootstrap_mean(candidate_error-reference_error, reference.subject_id,n_bootstrap,seed)
