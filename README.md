This repository provides the reproducible implementation associated with the manuscript:


Uncertainty-aware geochemical anomaly mapping using sequential Gaussian simulation and SHAP: application to the Jbel Saghro, Morocco

Repository: 

github.com/JanaMed25/cageo-s-26-02501-geochem-workflow

Workflow overview

The workflow combines:

1. quality control and normal-score transformation of multi-element geochemical data;

2. experimental and fitted variograms;

3. conditional Sequential Gaussian Simulation (SGS);

4. inverse transformation and uncertainty quantification;

5. Concentration–Area, local singularity and exploratory multifractal analysis;

6. spatially blocked Random Forest and Multi-Layer Perceptron classification;

7. SHAP-based global, local and ensemble explanations;

8. robustness classification into priority, resample and background areas;

9. spatial clustering of candidate zones; and

10. a normal-score geochemical correlation network

Repository structure
.
├── .github/workflows/tests.yml       # Automated tests and optional demo workflow
├── configs/demo.yaml                 # Reproducible demonstration configuration
├── data/
│   ├── README.md                     # Data access and restrictions
│   └── synthetic_geochem.csv         # Synthetic demonstration data
├── docs/
│   ├── METHODS.md                    # Implemented methods
│   ├── REPRODUCTION_GUIDE.md         # Installation and reproduction steps
│   ├── LIMITATIONS.md                # Scientific and data limitations
│   ├── OUTPUTS.md                    # Output-file description
│   └── workflow.mmd                  # Editable Mermaid workflow diagram
├── manuscript/
│   ├── CODE_AVAILABILITY_STATEMENT.md
│   ├── AUTHORSHIP_STATEMENT.md
│   ├── COMPETING_INTERESTS_STATEMENT.md
│   └── PROVENANCE.md
                                      # Outputs from the synthetic demonstration
├── scripts/make_synthetic_data.py    # Synthetic input generator
├── src/geochem_workflow/              # Python source package
├── tests/test_core.py                 # Unit tests
├── CITATION.cff
├── environment.yml
├── LICENSE
├── pyproject.toml
└── requirements.txt
