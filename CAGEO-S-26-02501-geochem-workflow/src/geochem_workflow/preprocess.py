"""Preprocessing helpers for positive geochemical concentrations."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata


@dataclass
class NormalScoreTransform:
    """Univariate empirical normal-score transform with an empirical inverse."""

    values_: np.ndarray | None = None
    scores_: np.ndarray | None = None

    def fit(self, values: np.ndarray | pd.Series) -> "NormalScoreTransform":
        x = np.asarray(values, dtype=float)
        x = x[np.isfinite(x)]
        if x.size < 2:
            raise ValueError("At least two finite observations are required for NST.")
        ranks = rankdata(x, method="average")
        probabilities = (ranks - 0.5) / x.size
        self.values_ = np.sort(x)
        self.scores_ = norm.ppf(np.clip(probabilities, 1e-7, 1 - 1e-7))
        order = np.argsort(x, kind="mergesort")
        self.scores_ = self.scores_[order]
        return self

    def transform(self, values: np.ndarray | pd.Series) -> np.ndarray:
        if self.values_ is None or self.scores_ is None:
            raise RuntimeError("Fit the transform before calling transform().")
        x = np.asarray(values, dtype=float)
        flat = x.ravel()
        out = np.full(flat.shape, np.nan, dtype=float)
        valid = np.isfinite(flat)
        xs = self.values_
        zs = self.scores_
        out[valid] = np.interp(flat[valid], xs, zs, left=zs[0], right=zs[-1])
        return out.reshape(x.shape)

    def inverse_transform(self, scores: np.ndarray | pd.Series) -> np.ndarray:
        if self.values_ is None or self.scores_ is None:
            raise RuntimeError("Fit the transform before calling inverse_transform().")
        z = np.asarray(scores, dtype=float)
        flat = z.ravel()
        out = np.full(flat.shape, np.nan, dtype=float)
        valid = np.isfinite(flat)
        unique_z, inverse = np.unique(self.scores_, return_inverse=True)
        mean_x = np.bincount(inverse, weights=self.values_) / np.bincount(inverse)
        out[valid] = np.interp(flat[valid], unique_z, mean_x, left=self.values_[0], right=self.values_[-1])
        return out.reshape(z.shape)


def centered_log_ratio(frame: pd.DataFrame, zero_replacement: float | None = None) -> pd.DataFrame:
    """Return row-wise CLR coordinates; zero replacement is explicit and auditable.

    A CLR vector is relative, not in the original concentration units. This
    function is for compositional predictors/correlation analysis, not a direct
    inverse transform for reconstructing absolute concentrations.
    """
    x = frame.astype(float).to_numpy(copy=True)
    if np.any(x[np.isfinite(x)] < 0):
        raise ValueError("CLR requires non-negative concentrations.")
    if zero_replacement is None:
        positive = x[np.isfinite(x) & (x > 0)]
        if positive.size == 0:
            raise ValueError("No positive values available to set a zero replacement.")
        zero_replacement = float(np.min(positive) * 0.5)
    if zero_replacement <= 0:
        raise ValueError("zero_replacement must be positive.")
    x[~np.isfinite(x)] = np.nan
    if np.isnan(x).any():
        raise ValueError("CLR input contains missing values; impute or remove them first.")
    x[x == 0] = zero_replacement
    log_x = np.log(x)
    clr = log_x - log_x.mean(axis=1, keepdims=True)
    return pd.DataFrame(clr, columns=frame.columns, index=frame.index)
