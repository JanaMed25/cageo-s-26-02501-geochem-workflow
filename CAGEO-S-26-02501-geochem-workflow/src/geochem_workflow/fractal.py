"""Exploratory concentration-area and multifractal calculations."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter


def concentration_area_threshold(values, min_points: int = 20) -> dict[str, float]:
    """Estimate a two-segment log(concentration)-log(area) breakpoint.

    The breakpoint is a data-driven screening threshold, not a validated ore
    threshold. Concentrations must be positive; zeros are omitted.
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    if x.size < max(2 * min_points, 8):
        raise ValueError("Too few positive observations for a stable C-A breakpoint.")
    thresholds = np.unique(x)
    area = np.array([(x >= t).sum() / x.size for t in thresholds], dtype=float)
    lx, ly = np.log(thresholds), np.log(area)
    order = np.argsort(lx)
    lx, ly, thresholds = lx[order], ly[order], thresholds[order]
    best = None
    min_levels = max(4, min_points)
    if len(lx) < 2 * min_levels:
        min_levels = max(2, len(lx) // 4)
    for split in range(min_levels, len(lx) - min_levels):
        # Require distinct concentration levels on both sides of the breakpoint.
        if np.unique(lx[:split]).size < 2 or np.unique(lx[split:]).size < 2:
            continue
        left_coef = np.polyfit(lx[:split], ly[:split], 1)
        right_coef = np.polyfit(lx[split:], ly[split:], 1)
        residual = np.square(ly[:split] - np.polyval(left_coef, lx[:split])).sum()
        residual += np.square(ly[split:] - np.polyval(right_coef, lx[split:])).sum()
        if best is None or residual < best[0]:
            best = (residual, split, left_coef, right_coef)
    if best is None:
        raise ValueError("Could not fit a two-segment C-A curve; check ties and sample size.")
    _, split, left, right = best
    return {
        "threshold": float(thresholds[split]),
        "breakpoint_log_concentration": float(lx[split]),
        "low_concentration_slope": float(left[0]),
        "high_concentration_slope": float(right[0]),
        "rss": float(best[0]),
        "n": int(x.size),
    }


def local_singularity_index(raster, window_sizes=(3, 5, 9, 15)) -> np.ndarray:
    """Estimate local mass-scaling exponent alpha from a positive 2-D raster.

    For a spatially uniform field in the raster interior, alpha is approximately
    2. This simple isotropic box-window estimator is an exploratory proxy; it is
    not a substitute for a study-specific implementation of Cheng (2007).
    """
    a = np.asarray(raster, dtype=float)
    if a.ndim != 2:
        raise ValueError("raster must be a 2-D array.")
    finite = np.isfinite(a)
    if not finite.any():
        raise ValueError("raster has no finite values.")
    if np.nanmin(a) < 0:
        raise ValueError("Singularity mass raster must be non-negative.")
    fill = np.nanmedian(a[finite])
    a = np.where(finite, a, fill)
    scales, log_mass = [], []
    for size in window_sizes:
        size = int(size)
        if size < 1 or size % 2 == 0:
            raise ValueError("window_sizes must contain positive odd integers.")
        mean_local = uniform_filter(a, size=size, mode="reflect")
        mass = np.maximum(mean_local * (size**2), np.finfo(float).tiny)
        scales.append(np.log(size))
        log_mass.append(np.log(mass))
    if len(scales) < 2:
        raise ValueError("At least two window sizes are required.")
    xx = np.asarray(scales)
    yy = np.stack(log_mass, axis=0)
    x0 = xx - xx.mean()
    return np.sum(x0[:, None, None] * (yy - yy.mean(axis=0)), axis=0) / np.sum(x0**2)


def multifractal_spectrum(raster, q_values=None, box_sizes=(2, 4, 8, 16)) -> dict[str, np.ndarray]:
    """Estimate a box-counting moment spectrum tau(q), alpha(q), and f(alpha).

    Output is exploratory and sensitive to grid extent, cell size, missing-data
    treatment, and the selected scales. Values are computed from non-overlapping
    blocks after cropping edges at each scale.
    """
    a = np.asarray(raster, dtype=float)
    if a.ndim != 2 or np.any(a[np.isfinite(a)] < 0):
        raise ValueError("raster must be a non-negative 2-D array.")
    if q_values is None:
        q_values = np.linspace(-2, 4, 13)
    q = np.asarray(q_values, dtype=float)
    if np.any(np.isclose(q, 1.0)):
        raise ValueError("q=1 is singular for this partition-function implementation; omit it.")
    tau_rows, eps = [], []
    for size in box_sizes:
        size = int(size)
        if size < 1:
            continue
        ny, nx = (a.shape[0] // size) * size, (a.shape[1] // size) * size
        if ny < size * 2 or nx < size * 2:
            continue
        block = a[:ny, :nx].reshape(ny // size, size, nx // size, size).sum(axis=(1, 3))
        mass = block[np.isfinite(block) & (block > 0)]
        if mass.size < 2:
            continue
        mass = mass / mass.sum()
        z = np.array([np.log(np.sum(mass**qi)) for qi in q])
        tau_rows.append(z)
        eps.append(size)
    if len(eps) < 2:
        raise ValueError("Need at least two usable box scales for the multifractal spectrum.")
    log_eps = np.log(1.0 / np.asarray(eps, dtype=float))
    tau = np.array([np.polyfit(log_eps, row, 1)[0] for row in np.asarray(tau_rows).T])
    alpha = np.gradient(tau, q)
    spectrum = q * alpha - tau
    return {"q": q, "tau": tau, "alpha": alpha, "f_alpha": spectrum, "box_sizes": np.asarray(eps)}
