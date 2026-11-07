"""Spatially blocked model validation, fitting, and optional SHAP explanations."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def spatial_groups(coords, block_size):
    """Assign samples to fixed square blocks in projected coordinate units."""
    xy = np.asarray(coords, dtype=float)
    if block_size <= 0:
        raise ValueError("block_size must be positive.")
    origin = np.nanmin(xy, axis=0)
    ij = np.floor((xy - origin) / float(block_size)).astype(int)
    return np.array([f"{i}_{j}" for i, j in ij], dtype=object)


def make_classifier(name: str, random_state: int = 2026):
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=400, min_samples_leaf=2, class_weight="balanced_subsample",
            n_jobs=-1, random_state=random_state,
        )
    if name == "mlp":
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True,
                          max_iter=500, random_state=random_state),
        )
    raise ValueError(f"Unknown algorithm {name!r}; use random_forest or mlp.")


def spatial_cross_validate(X, y, coords, n_splits=5, block_size=3000, random_state=2026,
                           algorithm="random_forest"):
    """Evaluate with non-overlapping spatial groups and stratified group folds."""
    x = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    groups = spatial_groups(coords, block_size)
    if np.unique(y).size != 2:
        raise ValueError("The proxy target must contain both binary classes 0 and 1.")
    if np.unique(groups).size < n_splits:
        raise ValueError(
            f"Only {np.unique(groups).size} spatial blocks; need at least {n_splits}. "
            "Reduce n_splits or block_size, and justify the choice."
        )
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rows = []
    for fold, (train, test) in enumerate(cv.split(x, y, groups), start=1):
        if np.unique(y[train]).size < 2:
            continue
        model = make_classifier(algorithm, random_state + fold)
        model.fit(x[train], y[train])
        pred = model.predict(x[test])
        prob = model.predict_proba(x[test])[:, 1]
        rows.append({
            "fold": fold,
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "n_test_blocks": int(np.unique(groups[test]).size),
            "accuracy": float(accuracy_score(y[test], pred)),
            "f1": float(f1_score(y[test], pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y[test], prob)) if np.unique(y[test]).size == 2 else np.nan,
        })
    if not rows:
        raise ValueError("No valid CV fold had both target classes in its training partition.")
    return pd.DataFrame(rows)


def positive_class_shap(model, X, algorithm: str, background, max_points: int = 500):
    """Return SHAP values for class 1; generic MLP SHAP uses a bounded sample."""
    try:
        import shap
    except ImportError as exc:
        raise ImportError("Install optional dependency with `pip install -e .[explain]`.") from exc
    x = np.asarray(X, dtype=float)
    if len(x) > max_points:
        rng = np.random.default_rng(2026)
        idx = np.sort(rng.choice(len(x), max_points, replace=False))
        x = x[idx]
    bg = np.asarray(background, dtype=float)
    if algorithm == "random_forest":
        explainer = shap.TreeExplainer(
            model, data=bg, feature_perturbation="interventional", model_output="probability"
        )
        values = np.asarray(explainer.shap_values(x))
        if values.ndim == 3:
            values = values[:, :, 1]
        return x, values
    # MLPs are model-agnostic here. Limit rows/background in the config; this can be slow.
    explainer = shap.PermutationExplainer(model.predict_proba, bg)
    explanation = explainer(x, max_evals=max(2 * x.shape[1] + 1, 2 * x.shape[1] + 1))
    values = np.asarray(explanation.values)
    if values.ndim == 3:
        values = values[:, :, 1]
    return x, values
