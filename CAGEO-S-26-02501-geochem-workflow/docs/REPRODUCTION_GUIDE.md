# Reproduction guide

## 1. Clean installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[explain,test]"
```

Alternatively use `environment.yml` with Conda.

## 2. Software test

```bash
pytest
```

The tests cover the normal-score round trip, CLR transformation, concentration–area threshold, local singularity dimensions, multifractal output shapes, spatial block assignment and SHAP aggregation logic.

## 3. Synthetic end-to-end run

```bash
python scripts/make_synthetic_data.py
geochem-workflow --config configs/demo.yaml
```

The synthetic run is designed to test the complete code path. Because it performs conditional SGS and SHAP, execution time depends on the machine and installed versions.

## 4. Real study data

Create a study-specific YAML file. At minimum, set `input_csv`, `coordinates`, `elements`, `label_column` or `proxy_element`, `output_dir`, `model.block_size`, `simulation.realizations`, `grid_spacing` and the correct projected coordinate units. Do not use the synthetic CSV as scientific evidence.

## 5. Main outputs

The workflow writes `spatial_cv_metrics.csv`, SGS realizations, variogram diagnostics, concentration–area thresholds, singularity arrays, multifractal spectra, per-cell anomaly probabilities, SHAP summaries, robustness classes, candidate-zone clusters, network nodes/edges and `run_metadata.json`.

## 6. Reproducibility record

Archive the exact YAML configuration, commit hash, Python version, package versions, random seeds, coordinate reference system, input-data checksum and output manifest with each manuscript release.
