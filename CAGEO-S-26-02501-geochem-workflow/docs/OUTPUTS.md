# Output manifest

A run directory contains the following output families:

- `spatial_cv_metrics.csv`: fold-wise accuracy, F1 and ROC-AUC for each classifier;
- `variograms/`: experimental and fitted variograms by element;
- `concentration_area_thresholds.csv`: C–A breakpoint estimates;
- `fractal/`: singularity arrays, multifractal spectra and scale tables;
- `*_probability_*.csv`: per-cell probabilities and uncertainty summaries;
- `*_shap_*.csv`: mean and variance of sampled SHAP contributions;
- `*_clusters.csv`: spatial candidate-zone clusters;
- `network_nodes.csv`, `network_edges.csv` and `geochemical_correlation_network.graphml`: correlation-network outputs;
- `run_metadata.json`: configuration, seeds and run provenance.
