# Code availability statement

The source code, configuration files, synthetic demonstration data, unit tests and documentation for the geochemical anomaly-mapping workflow are available at **[INSERT GITHUB REPOSITORY URL]**, release **[INSERT VERSION OR DOI]**. The repository implements the preprocessing, conditional sequential Gaussian simulation, variogram fitting, concentration–area analysis, exploratory singularity and multifractal calculations, spatial-block Random Forest and MLP classification, SHAP aggregation, robustness classification and correlation-network analysis described in the manuscript.

The repository includes a reproducible synthetic demonstration. The study-specific geochemical data are not redistributed unless permitted by the data owner; authorized users should place the input CSV in the `data/` directory and use the study-specific YAML configuration. The release records the Python environment, random seeds, spatial-block settings and output manifest.
