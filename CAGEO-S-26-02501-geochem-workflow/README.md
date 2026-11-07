# Reproducible Geochemical Anomaly Mapping Workflow

Repository prepared from `geochem.workflow.zip` for manuscript **CAGEO-S-26-02501(1)** submitted to *Computers & Geosciences*.

## Scope

This repository contains the actual modular workflow supplied in the archive. It combines normal-score preprocessing, conditional sequential Gaussian simulation (SGS), variogram fitting, concentration–area breakpoint analysis, exploratory local singularity and multifractal analysis, spatially blocked Random Forest and MLP classification, SHAP explanations, robustness aggregation and a geochemical correlation network.

The included CSV is explicitly synthetic and is provided only for a reproducible software demonstration. It must not be used as evidence for a real mineral occurrence. For the submitted study, place the authorized project CSV in `data/` and use a study-specific YAML configuration.

## Repository structure

```text
.
├── README.md
├── LICENSE
├── CITATION.cff
├── pyproject.toml
├── requirements.txt
├── environment.yml
├── configs/
│   └── demo.yaml
├── data/
│   └── synthetic_geochem.csv
├── docs/
│   ├── METHODS.md
│   ├── REPRODUCTION_GUIDE.md
│   ├── LIMITATIONS.md
│   └── workflow.png
├── figures/
├── results/
├── scripts/
│   └── make_synthetic_data.py
├── src/geochem_workflow/
│   ├── cli.py
│   ├── preprocess.py
│   ├── geostat.py
│   ├── fractal.py
│   ├── models.py
│   ├── robustness.py
│   └── network.py
└── tests/
    └── test_core.py
```

## Installation

Python 3.10 or later is required. A standard installation is:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[explain,test]"
```

The workflow uses GStatSim for conditional SGS. If the package is unavailable on the target platform, install the dependencies listed in `requirements.txt` and consult the GStatSim documentation.

## Synthetic demonstration

Run the unit tests:

```bash
pytest
```

Run the complete demonstration:

```bash
python scripts/make_synthetic_data.py
geochem-workflow --config configs/demo.yaml
```

The run writes outputs under `results/demo/`, including spatial-CV metrics, SGS realizations, variograms, concentration–area thresholds, singularity arrays, multifractal spectra, SHAP summaries, robustness classifications, candidate clusters and correlation-network files.

Without installing the package, the CLI can be invoked with:

```bash
PYTHONPATH=src python -m geochem_workflow.cli --config configs/demo.yaml
```

## Input-data requirements

The configuration must identify projected `Easting` and `Northing` coordinates, positive concentration columns and either a binary `label_column` or a `proxy_element` from which a concentration–area anomaly proxy is derived. Coordinates must use a consistent linear unit compatible with `block_size`, `grid_spacing` and `search_radius`.

The source archive does not contain the confidential or study-specific dataset from the submitted manuscript. Do not commit restricted geochemical data to GitHub. Instead, document the authorized access procedure in `data/README.md` and keep the data file out of version control.

## Reproducibility and leakage control

If an anomaly label is derived from an element, that element is automatically excluded from the model predictors. If a supplied label was constructed from several elements, list every contributing element under `exclude_from_predictors`. Report the random seeds, coordinate reference system, block size, number of folds, simulation settings and exact configuration file with each manuscript release.

## Interpretation

The generated products are exploratory screening outputs and candidate-zone hypotheses. They are not mineral-resource estimates, drilling recommendations or causal evidence of a shared mineralizing process. The singularity and multifractal functions are explicitly exploratory estimators and require study-specific validation of scale selection, edge effects and geological interpretation.

## Citation and release

Replace the placeholder repository URL and author metadata in `CITATION.cff`. Before submission, create a tagged GitHub release and archive it with Zenodo or an equivalent service to obtain a DOI. Report the release tag, DOI, Python version, dependency versions, random seeds, CRS and data-access conditions in the manuscript.
