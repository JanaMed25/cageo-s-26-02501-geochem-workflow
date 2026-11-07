"""Generate an explicitly synthetic, reproducible geochemical-like CSV."""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(2026)
n = 240
x = rng.uniform(0, 20_000, n)
y = rng.uniform(0, 18_000, n)
latent = np.sin(x / 5000) + np.cos(y / 4300) + rng.normal(0, 0.35, n)
cu = np.exp(3.0 + 0.55 * latent + rng.normal(0, 0.20, n))
zn = np.exp(2.6 + 0.43 * latent + rng.normal(0, 0.23, n))
pb = np.exp(2.1 + 0.34 * latent + rng.normal(0, 0.25, n))
# Synthetic label from a latent field, not from a predictor's threshold.
proxy = (latent > np.quantile(latent, 0.78)).astype(int)
out = Path(__file__).resolve().parents[1] / "data" / "synthetic_geochem.csv"
out.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame({"Easting": x, "Northing": y, "Cu": cu, "Zn": zn, "Pb": pb,
              "anomaly_proxy": proxy}).to_csv(out, index=False)
print(out)
