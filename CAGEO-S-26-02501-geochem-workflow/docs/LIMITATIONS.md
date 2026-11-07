# Limitations and publication conditions

The included dataset is synthetic. It is not the Tiouit–Ikneewin/Jbel Saghro study dataset and must not be used to support geological or exploration claims.

The SGS implementation uses isotropic omnidirectional variograms and local ordinary-kriging neighborhoods. Directional anisotropy, trends, geological domains, declustering and non-stationarity require study-specific analysis.

The singularity and multifractal routines are labelled exploratory in the source code. They should not be presented as a fully validated geological inference without scale sensitivity, edge-effect diagnostics and comparison with the exact method cited in the manuscript.

The block-based spatial cross-validation tests transfer to held-out square blocks but does not guarantee independence, calibration or transfer to a different catchment. Report block size, fold composition and fold-wise metrics.

The repository is ready for a journal code-availability submission only after the authors replace synthetic input references with the authorized study-data access procedure, insert the exact study YAML configuration, run the complete workflow successfully and archive the resulting release.
