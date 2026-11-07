# Methods implemented in this repository

The workflow first reads projected sample coordinates and strictly positive element concentrations. Each element is transformed with an empirical normal-score transform for variogram modelling and conditional sequential Gaussian simulation, then inverse-transformed to the concentration scale.

An omnidirectional experimental variogram is estimated from reproducibly sampled pairs and fitted with a spherical, exponential or Gaussian isotropic model. GStatSim is then used to generate conditional SGS realizations on a regular grid.

The fractal branch calculates a reproducible two-segment concentration–area breakpoint, a local box-window singularity index and an exploratory moment-based multifractal spectrum. These outputs require study-specific validation of scales, edge effects and geological interpretation.

The classification branch uses Random Forest and/or MLP models with non-overlapping square spatial blocks and `StratifiedGroupKFold`. SHAP explanations are calculated on an explicit subset of grid cells, and their mean and variance are aggregated across realizations. A robustness classifier labels cells as priority, resample or background according to configured probability, SHAP and coherence thresholds.

The network branch transforms the element values to normal scores or CLR-normal scores, computes positive Pearson correlations and retains edges after a Benjamini–Hochberg false-discovery-rate filter. Correlation is descriptive and is not evidence of a common mineralizing process.
