"""Uncertainty-aware cell-level aggregation and candidate-zone labels."""
from __future__ import annotations

import numpy as np


def aggregate_shap(shap_by_realization):
    """Return mean and variance over realizations for a 3-D (realization, cell, feature) array."""
    values = np.asarray(shap_by_realization, dtype=float)
    if values.ndim != 3:
        raise ValueError("Expected SHAP values with shape (realization, cell, feature).")
    return np.nanmean(values, axis=0), np.nanvar(values, axis=0, ddof=1 if len(values) > 1 else 0)


def classify_robustness(probability_by_realization, shap_mean, shap_variance,
                        probability_threshold=0.75, shap_mean_quantile=0.75,
                        shap_sd_quantile=0.50):
    """Label cells as priority, resample, or background using explicit quantiles.

    Priority requires high anomaly probability, high positive mean SHAP contribution,
    and low SHAP standard deviation. High-probability cells with unstable or weak
    attribution are marked for resampling. Labels are decision-support hypotheses.
    """
    p = np.asarray(probability_by_realization, dtype=float)
    mean = np.asarray(shap_mean, dtype=float)
    variance = np.asarray(shap_variance, dtype=float)
    if p.ndim != 2 or mean.shape != variance.shape or p.shape[1] != mean.shape[0]:
        raise ValueError("Probability and SHAP arrays have inconsistent shapes.")
    p_mean = np.nanmean(p, axis=0)
    p_sd = np.nanstd(p, axis=0, ddof=1 if p.shape[0] > 1 else 0)
    mean_contribution = np.nanmean(mean, axis=1)
    sd_contribution = np.sqrt(np.nanmean(variance, axis=1))
    positive = mean_contribution[mean_contribution > 0]
    mean_cut = np.quantile(positive, shap_mean_quantile) if positive.size else np.inf
    sd_cut = np.quantile(sd_contribution[np.isfinite(sd_contribution)], shap_sd_quantile)
    high_prob = p_mean >= probability_threshold
    stable = (mean_contribution >= mean_cut) & (sd_contribution <= sd_cut)
    label = np.full(len(p_mean), "background", dtype=object)
    label[high_prob & ~stable] = "resample"
    label[high_prob & stable] = "priority"
    return {
        "probability_mean": p_mean,
        "probability_sd": p_sd,
        "shap_mean_contribution": mean_contribution,
        "shap_sd_contribution": sd_contribution,
        "robustness_class": label,
        "shap_mean_threshold": float(mean_cut),
        "shap_sd_threshold": float(sd_cut),
    }
