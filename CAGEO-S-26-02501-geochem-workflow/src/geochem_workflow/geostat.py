"""Normal-score geostatistics and conditional sequential Gaussian simulation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.spatial.distance import pdist

from .preprocess import NormalScoreTransform


class _ScalarNormalRNG:
    """Narrow NumPy-2 compatibility adapter for GStatSim's size-1 assignment."""

    def __init__(self, seed):
        self._generator = np.random.default_rng(seed)

    def shuffle(self, values):
        return self._generator.shuffle(values)

    def normal(self, loc=0.0, scale=1.0, size=None):
        sample = self._generator.normal(loc, scale, size)
        if size == 1:
            return float(np.asarray(sample).reshape(-1)[0])
        return sample


def empirical_variogram(coords, values, max_lag_fraction=0.5, n_bins=12, max_pairs=100_000, seed=2026):
    """Compute an omnidirectional method-of-moments experimental variogram."""
    xy = np.asarray(coords, dtype=float)
    z = np.asarray(values, dtype=float)
    valid = np.isfinite(xy).all(axis=1) & np.isfinite(z)
    xy, z = xy[valid], z[valid]
    if len(z) < 6:
        raise ValueError("At least six valid samples are required for variogram estimation.")
    rng = np.random.default_rng(seed)
    n_pairs = len(z) * (len(z) - 1) // 2
    if n_pairs > max_pairs:
        # Subsample pairs reproducibly rather than allocating the full O(n^2) set.
        i, j = np.triu_indices(len(z), k=1)
        keep = rng.choice(len(i), size=max_pairs, replace=False)
        i, j = i[keep], j[keep]
        lag = np.linalg.norm(xy[i] - xy[j], axis=1)
        semivar = 0.5 * np.square(z[i] - z[j])
    else:
        lag = pdist(xy)
        dz = pdist(z[:, None])
        semivar = 0.5 * np.square(dz)
    positive = lag > 0
    lag, semivar = lag[positive], semivar[positive]
    max_lag = float(np.max(lag) * max_lag_fraction)
    edges = np.linspace(0, max_lag, n_bins + 1)
    centers, gamma, counts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (lag >= lo) & (lag < hi)
        if mask.sum() >= 3:
            centers.append(float(lag[mask].mean()))
            gamma.append(float(semivar[mask].mean()))
            counts.append(int(mask.sum()))
    if len(centers) < 4:
        raise ValueError("Too few populated lag bins; change max_lag_fraction or n_bins.")
    return np.asarray(centers), np.asarray(gamma), np.asarray(counts)


def fit_variogram(coords, values, model="Spherical", max_lag_fraction=0.5, n_bins=12,
                  max_pairs=100_000, seed=2026):
    """Fit isotropic spherical/exponential/Gaussian model; return GStatSim params."""
    name = model.capitalize()
    if name not in {"Spherical", "Exponential", "Gaussian"}:
        raise ValueError("model must be Spherical, Exponential, or Gaussian.")
    h, empirical, counts = empirical_variogram(
        coords, values, max_lag_fraction, n_bins, max_pairs, seed
    )
    sill0 = max(float(np.nanvar(values)), float(np.max(empirical)), 1e-6)

    def theoretical(params):
        nugget, structured, range_ = params
        r = np.maximum(h / range_, 0)
        if name == "Spherical":
            shape = np.where(r < 1, 1.5 * r - 0.5 * r**3, 1.0)
        elif name == "Exponential":
            shape = 1.0 - np.exp(-3.0 * r)
        else:
            shape = 1.0 - np.exp(-3.0 * r**2)
        return nugget + structured * shape

    init = np.array([max(0.0, empirical[0] * 0.25), sill0, max(h)])
    lower = [0.0, 1e-8, max(np.min(h) * 0.1, 1e-8)]
    upper = [sill0, sill0 * 5 + 1e-6, max(h) * 10]
    fit = least_squares(lambda p: np.sqrt(counts) * (theoretical(p) - empirical), init,
                        bounds=(lower, upper), max_nfev=3000)
    nugget, structured, range_ = fit.x
    total_sill = float(nugget + structured)
    return {
        "parameters": [0.0, float(nugget), float(range_), float(range_), total_sill, name],
        "lags": h,
        "experimental": empirical,
        "fitted": theoretical(fit.x),
        "pair_counts": counts,
    }


def simulate_element(samples: pd.DataFrame, grid_xy: np.ndarray, x_col: str, y_col: str,
                     value_col: str, settings: dict, seed: int,
                     fit_seed: int | None = None) -> tuple[np.ndarray, dict]:
    """Fit NST and run one conditional SGS realization with GStatSim."""
    from gstatsim import Interpolation

    raw = samples[value_col].to_numpy(dtype=float)
    if np.any(raw[np.isfinite(raw)] <= 0):
        raise ValueError(f"{value_col}: SGS currently requires strictly positive concentrations.")
    nst = NormalScoreTransform().fit(raw)
    scores = nst.transform(raw)
    xy = samples[[x_col, y_col]].to_numpy(dtype=float)
    fit = fit_variogram(
        xy, scores,
        model=settings.get("variogram_model", "Spherical"),
        max_lag_fraction=settings.get("max_lag_fraction", 0.5),
        n_bins=settings.get("lag_bins", 12),
        max_pairs=settings.get("max_pairs", 100_000),
        seed=seed if fit_seed is None else fit_seed,
    )
    conditioning = pd.DataFrame({"X": xy[:, 0], "Y": xy[:, 1], "Z": scores})
    # GStatSim 1.2 passes rng.normal(..., 1) directly into a scalar cell. NumPy 2
    # rejects this 1-element array assignment. Adapt only its RNG factory for this
    # synchronous call, restoring the original factory even if simulation fails.
    import gstatsim
    original_factory = gstatsim.get_random_generator
    gstatsim.get_random_generator = lambda value: _ScalarNormalRNG(value)
    try:
        sim_z = Interpolation.okrige_sgs(
            np.asarray(grid_xy, dtype=float), conditioning, "X", "Y", "Z",
            int(settings.get("neighbors", 16)), fit["parameters"],
            float(settings.get("search_radius", np.inf)), seed=int(seed), quiet=True,
        )
    finally:
        gstatsim.get_random_generator = original_factory
    sim_raw = nst.inverse_transform(np.asarray(sim_z, dtype=float))
    diagnostics = {key: value for key, value in fit.items() if key != "parameters"}
    diagnostics["parameters"] = fit["parameters"]
    return sim_raw, diagnostics
