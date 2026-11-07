"""Command-line entry point for the reproducible geochemical workflow."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import norm
from sklearn.cluster import DBSCAN

from .fractal import concentration_area_threshold, local_singularity_index, multifractal_spectrum
from .geostat import simulate_element
from .models import make_classifier, positive_class_shap, spatial_cross_validate
from .network import correlation_network
from .preprocess import NormalScoreTransform, centered_log_ratio
from .robustness import aggregate_shap, classify_robustness


def _write_json(path: Path, data):
    def convert(value):
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(type(value).__name__)
    path.write_text(json.dumps(data, indent=2, default=convert) + "\n", encoding="utf-8")


def _model_features(values, feature_names, transform, zero_replacement):
    frame = pd.DataFrame(np.asarray(values, dtype=float), columns=feature_names)
    if transform == "raw":
        return frame.to_numpy(dtype=float)
    if transform == "clr":
        return centered_log_ratio(frame, zero_replacement=zero_replacement).to_numpy(dtype=float)
    raise ValueError("model.transform must be 'raw' or 'clr'.")


def run(config_path: str) -> Path:
    config_path = Path(config_path).resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base = config_path.parent.parent
    input_path = Path(cfg["input_csv"])
    if not input_path.is_absolute():
        input_path = base / input_path
    output = Path(cfg.get("output_dir", "outputs"))
    if not output.is_absolute():
        output = base / output
    output.mkdir(parents=True, exist_ok=True)

    x_col, y_col = cfg["coordinates"]
    elements = list(cfg["elements"])
    required = [x_col, y_col, *elements]
    frame = pd.read_csv(input_path)
    absent = [c for c in required if c not in frame]
    if absent:
        raise ValueError(f"Missing required input columns: {absent}")
    label_col = cfg.get("label_column")
    if label_col and label_col in frame:
        frame["_target"] = pd.to_numeric(frame[label_col], errors="coerce")
    elif cfg.get("proxy_element"):
        proxy = cfg["proxy_element"]
        if proxy not in elements:
            raise ValueError("proxy_element must be listed in elements.")
        ca = concentration_area_threshold(
            frame[proxy].to_numpy(), int(cfg.get("fractal", {}).get("ca_min_points", 20))
        )
        frame["_target"] = (frame[proxy] >= ca["threshold"]).astype(int)
        cfg["_derived_proxy"] = ca
        cfg["exclude_from_predictors"] = sorted(set(cfg.get("exclude_from_predictors", [])) | {proxy})
    else:
        raise ValueError(
            f"Provide binary label column {label_col!r} or set proxy_element in the YAML config."
        )
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=[*required, "_target"]).copy()
    frame["_target"] = frame["_target"].astype(int)
    if not set(frame["_target"].unique()).issubset({0, 1}) or frame["_target"].nunique() != 2:
        raise ValueError("Target must contain both binary classes encoded as 0 and 1.")
    if any((frame[e] <= 0).any() for e in elements):
        raise ValueError("All SGS input concentrations must be > 0; define a documented zero policy.")

    excluded = set(cfg.get("exclude_from_predictors", []))
    features = [e for e in elements if e not in excluded]
    if not features:
        raise ValueError("No predictor remains after exclude_from_predictors.")
    coords = frame[[x_col, y_col]].to_numpy(dtype=float)
    model_cfg = cfg.get("model", {})
    model_transform = model_cfg.get("transform", "raw")
    clr_zero = model_cfg.get("clr_zero_replacement")
    X = _model_features(frame[features].to_numpy(dtype=float), features, model_transform, clr_zero)
    y = frame["_target"].to_numpy(dtype=int)
    seed = int(model_cfg.get("random_state", 2026))
    cv_frames = []
    for algorithm in model_cfg.get("algorithms", ["random_forest"]):
        cv = spatial_cross_validate(
            X, y, coords,
            n_splits=int(model_cfg.get("folds", 5)),
            block_size=float(model_cfg.get("block_size", 3000)),
            random_state=seed,
            algorithm=algorithm,
        )
        cv.insert(0, "algorithm", algorithm)
        cv_frames.append(cv)
    pd.concat(cv_frames, ignore_index=True).to_csv(output / "spatial_cv_metrics.csv", index=False)

    spacing = float(cfg.get("grid_spacing", 500))
    if spacing <= 0:
        raise ValueError("grid_spacing must be positive.")
    gx = np.arange(np.floor(coords[:, 0].min() / spacing) * spacing,
                   np.ceil(coords[:, 0].max() / spacing) * spacing + spacing * 0.5, spacing)
    gy = np.arange(np.floor(coords[:, 1].min() / spacing) * spacing,
                   np.ceil(coords[:, 1].max() / spacing) * spacing + spacing * 0.5, spacing)
    xx, yy = np.meshgrid(gx, gy)
    grid_xy = np.column_stack([xx.ravel(), yy.ravel()])
    if len(grid_xy) > int(cfg.get("max_grid_cells", 100_000)):
        raise ValueError("Grid exceeds max_grid_cells; increase grid_spacing or set a documented limit.")

    sim_cfg = cfg.get("simulation", {})
    n_real = int(sim_cfg.get("realizations", 20))
    if n_real < 2:
        raise ValueError("At least two realizations are required to estimate uncertainty.")
    sim = np.empty((n_real, len(grid_xy), len(elements)), dtype=float)
    variogram_summary = {}
    variogram_dir = output / "variograms"
    variogram_dir.mkdir(exist_ok=True)
    for j, element in enumerate(elements):
        for r in range(n_real):
            seed_r = int(sim_cfg.get("seed", 2026)) + r * 1009 + j * 9176
            fit_seed = int(sim_cfg.get("seed", 2026)) + j * 9176
            simulated, diagnostics = simulate_element(
                frame, grid_xy, x_col, y_col, element, sim_cfg, seed_r, fit_seed=fit_seed
            )
            sim[r, :, j] = simulated
            if r == 0:
                pd.DataFrame({
                    "lag": diagnostics["lags"],
                    "experimental_semivariance": diagnostics["experimental"],
                    "fitted_semivariance": diagnostics["fitted"],
                    "pair_count": diagnostics["pair_counts"],
                }).to_csv(variogram_dir / f"{element}_variogram.csv", index=False)
            variogram_summary[f"{element}_realization_{r + 1}"] = {
                "parameters": diagnostics["parameters"],
                "mean_experimental_semivariance": float(np.mean(diagnostics["experimental"])),
            }
        print(f"SGS complete: {element} ({n_real} realizations)", flush=True)

    # Characterize each simulated concentration field using the ensemble mean.
    fractal_cfg = cfg.get("fractal", {})
    ca_records = []
    fractal_dir = output / "fractal"
    fractal_dir.mkdir(exist_ok=True)
    for j, element in enumerate(elements):
        ca_records.append({"element": element, **concentration_area_threshold(
            frame[element].to_numpy(), int(fractal_cfg.get("ca_min_points", 20))
        )})
        mean_map = sim[:, :, j].mean(axis=0).reshape(xx.shape)
        windows = fractal_cfg.get("singularity_window_sizes", [3, 5, 9, 15])
        singularity = local_singularity_index(mean_map, windows)
        np.save(fractal_dir / f"{element}_singularity_alpha.npy", singularity)
        if element in fractal_cfg.get("elements", []):
            q = np.linspace(
                float(fractal_cfg.get("multifractal_q_min", -2)),
                float(fractal_cfg.get("multifractal_q_max", 4)),
                int(fractal_cfg.get("multifractal_q_count", 13)),
            )
            q = q[~np.isclose(q, 1.0)]
            spectrum = multifractal_spectrum(mean_map, q_values=q)
            pd.DataFrame({key: value for key, value in spectrum.items() if key != "box_sizes"}).to_csv(
                fractal_dir / f"{element}_multifractal_spectrum.csv", index=False
            )
            pd.DataFrame({"box_size_cells": spectrum["box_sizes"]}).to_csv(
                fractal_dir / f"{element}_multifractal_scales.csv", index=False
            )
    pd.DataFrame(ca_records).to_csv(output / "concentration_area_thresholds.csv", index=False)
    ca_thresholds = np.array([next(row["threshold"] for row in ca_records if row["element"] == e)
                              for e in elements], dtype=float)
    multielement_coherence = np.mean(sim >= ca_thresholds[None, None, :], axis=2)
    coherence_mean = multielement_coherence.mean(axis=0)

    # The association network is based on per-element normal scores, as in the figure text.
    network_cfg = cfg.get("network", {})
    network_transform = network_cfg.get("transform", "normal_score")
    if network_transform == "normal_score":
        network_values = frame[elements].copy()
    elif network_transform == "clr_normal_score":
        network_values = centered_log_ratio(
            frame[elements], zero_replacement=network_cfg.get("clr_zero_replacement")
        )
    else:
        raise ValueError("network.transform must be 'normal_score' or 'clr_normal_score'.")
    ns = {}
    for element in elements:
        nst = NormalScoreTransform().fit(network_values[element].to_numpy())
        ns[element] = nst.transform(network_values[element].to_numpy())
    ns_frame = pd.DataFrame(ns)
    nodes, edges, graph = correlation_network(
        ns_frame,
        float(cfg.get("network", {}).get("correlation_threshold", 0.5)),
        float(cfg.get("network", {}).get("fdr_q", 0.05)),
    )
    nodes.to_csv(output / "network_nodes.csv", index=False)
    edges.to_csv(output / "network_edges.csv", index=False)
    import networkx as nx
    nx.write_graphml(graph, output / "geochemical_correlation_network.graphml")

    # Explain a reproducible subset of cells, shared across realizations.
    explain_cfg = model_cfg
    n_shap = min(int(explain_cfg.get("shap_points", 500)), len(grid_xy))
    rng = np.random.default_rng(seed)
    explained_idx = np.sort(rng.choice(len(grid_xy), n_shap, replace=False))
    n_background = min(int(explain_cfg.get("shap_background", 80)), len(X))
    background = X[np.sort(rng.choice(len(X), n_background, replace=False))]
    all_map_frames = []
    for algorithm in model_cfg.get("algorithms", ["random_forest"]):
        prob_by_real = np.empty((n_real, len(grid_xy)), dtype=float)
        shap_by_real = np.empty((n_real, n_shap, len(features)), dtype=float)
        for r in range(n_real):
            realized_features = _model_features(
                pd.DataFrame(sim[r, :, :], columns=elements)[features].to_numpy(),
                features, model_transform, clr_zero,
            )
            model = make_classifier(algorithm, seed)
            model.fit(X, y)
            class_one = int(np.flatnonzero(model.classes_ == 1)[0])
            prob_by_real[r] = model.predict_proba(realized_features)[:, class_one]
            _, shap_values = positive_class_shap(
                model, realized_features[explained_idx], algorithm, background,
                max_points=n_shap,
            )
            shap_by_real[r] = shap_values
            print(f"Model complete: {algorithm}, realization {r + 1}/{n_real}", flush=True)
        shap_mean, shap_var = aggregate_shap(shap_by_real)
        robust = classify_robustness(
            prob_by_real[:, explained_idx], shap_mean, shap_var,
            float(cfg.get("robustness", {}).get("probability_threshold", 0.75)),
            float(cfg.get("robustness", {}).get("shap_mean_quantile", 0.75)),
            float(cfg.get("robustness", {}).get("shap_sd_quantile", 0.50)),
        )
        minimum_coherence = float(cfg.get("robustness", {}).get("minimum_coherence", 0.5))
        robust_class = np.asarray(robust["robustness_class"], dtype=object)
        weak_coherence = (robust_class == "priority") & (
            coherence_mean[explained_idx] < minimum_coherence
        )
        robust_class[weak_coherence] = "resample"
        result = pd.DataFrame({
            "x": grid_xy[:, 0], "y": grid_xy[:, 1],
            "probability_mean": prob_by_real.mean(axis=0),
            "probability_sd": prob_by_real.std(axis=0, ddof=1 if n_real > 1 else 0),
            "multielement_coherence": coherence_mean,
            "robustness_class": "not_explained",
            "shap_mean_contribution": np.nan,
            "shap_sd_contribution": np.nan,
            "cluster_id": -1,
        })
        result.loc[explained_idx, "robustness_class"] = robust_class
        result.loc[explained_idx, "shap_mean_contribution"] = robust["shap_mean_contribution"]
        result.loc[explained_idx, "shap_sd_contribution"] = robust["shap_sd_contribution"]
        priority_idx = explained_idx[robust_class == "priority"]
        if len(priority_idx):
            clustering = DBSCAN(
                eps=float(cfg.get("robustness", {}).get("cluster_epsilon", spacing * 2.0)),
                min_samples=int(cfg.get("robustness", {}).get("cluster_min_samples", 3)),
            ).fit(grid_xy[priority_idx])
            result.loc[priority_idx, "cluster_id"] = clustering.labels_
            cluster_rows = []
            for cluster_id in sorted(set(clustering.labels_) - {-1}):
                members = priority_idx[clustering.labels_ == cluster_id]
                cluster_rows.append({
                    "algorithm": algorithm,
                    "cluster_id": int(cluster_id),
                    "n_cells": int(len(members)),
                    "mean_probability": float(result.loc[members, "probability_mean"].mean()),
                    "mean_multielement_coherence": float(
                        result.loc[members, "multielement_coherence"].mean()
                    ),
                    "mean_shap_contribution": float(
                        result.loc[members, "shap_mean_contribution"].mean()
                    ),
                })
            pd.DataFrame(cluster_rows, columns=[
                "algorithm", "cluster_id", "n_cells", "mean_probability",
                "mean_multielement_coherence", "mean_shap_contribution",
            ]).to_csv(output / f"candidate_zones_{algorithm}.csv", index=False)
        else:
            pd.DataFrame(columns=["algorithm", "cluster_id", "n_cells", "mean_probability",
                                  "mean_multielement_coherence", "mean_shap_contribution"]).to_csv(
                output / f"candidate_zones_{algorithm}.csv", index=False
            )
        result.insert(0, "algorithm", algorithm)
        result.to_csv(output / f"decision_support_{algorithm}.csv", index=False)
        shap_rows = []
        for k, feature in enumerate(features):
            shap_rows.append({"feature": feature, "mean_shap": float(shap_mean[:, k].mean()),
                              "variance_shap": float(shap_var[:, k].mean())})
        pd.DataFrame(shap_rows).to_csv(output / f"shap_summary_{algorithm}.csv", index=False)
        all_map_frames.append(result)

    _write_json(output / "run_metadata.json", {
        "config": cfg,
        "input_file": str(input_path),
        "n_samples": len(frame),
        "n_grid_cells": len(grid_xy),
        "n_realizations": n_real,
        "predictor_features": features,
        "note": "Priority/resample classes are screening hypotheses, not validated drilling targets.",
        "variograms": variogram_summary,
    })
    return output


def main():
    parser = argparse.ArgumentParser(description="Run the geochemical anomaly mapping workflow.")
    parser.add_argument("--config", required=True, help="Path to a YAML workflow configuration.")
    args = parser.parse_args()
    out = run(args.config)
    print(f"Workflow complete. Outputs: {out}")


if __name__ == "__main__":
    main()
