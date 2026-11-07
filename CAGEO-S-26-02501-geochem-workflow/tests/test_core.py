import numpy as np
import pandas as pd

from geochem_workflow.fractal import concentration_area_threshold, local_singularity_index, multifractal_spectrum
from geochem_workflow.models import spatial_groups
from geochem_workflow.preprocess import NormalScoreTransform, centered_log_ratio
from geochem_workflow.robustness import aggregate_shap, classify_robustness


def test_normal_score_transform_round_trip():
    values = np.array([1.0, 2.0, 2.0, 4.0, 7.0, 10.0])
    transform = NormalScoreTransform().fit(values)
    recovered = transform.inverse_transform(transform.transform(values))
    np.testing.assert_allclose(recovered, values, atol=1e-8)


def test_clr_rows_sum_to_zero_with_zero_replacement():
    clr = centered_log_ratio(pd.DataFrame({"a": [1, 0], "b": [2, 3], "c": [4, 5]}))
    np.testing.assert_allclose(clr.sum(axis=1), 0.0, atol=1e-12)


def test_ca_threshold_is_positive_and_finite():
    rng = np.random.default_rng(8)
    values = np.exp(rng.normal(size=200))
    result = concentration_area_threshold(values, min_points=15)
    assert result["threshold"] > 0
    assert np.isfinite(result["rss"])


def test_uniform_raster_has_dimension_two_interior():
    raster = np.ones((64, 64))
    alpha = local_singularity_index(raster, (3, 5, 9, 15))
    np.testing.assert_allclose(alpha[16:-16, 16:-16], 2.0, atol=1e-10)


def test_multifractal_output_shapes():
    rng = np.random.default_rng(5)
    raster = rng.lognormal(size=(64, 64))
    result = multifractal_spectrum(raster, q_values=np.array([-2, 0, 2, 3]), box_sizes=(2, 4, 8, 16))
    assert result["alpha"].shape == (4,)
    assert result["f_alpha"].shape == (4,)


def test_spatial_groups_share_label_within_block():
    groups = spatial_groups(np.array([[10, 10], [20, 20], [4000, 10]]), block_size=1000)
    assert groups[0] == groups[1]
    assert groups[0] != groups[2]


def test_shap_aggregation_and_labels():
    values = np.ones((3, 4, 2))
    mean, variance = aggregate_shap(values)
    np.testing.assert_allclose(mean, 1)
    np.testing.assert_allclose(variance, 0)
    result = classify_robustness(
        np.array([[0.9, 0.1, 0.1, 0.1], [0.9, 0.1, 0.1, 0.1]]), mean, variance,
        probability_threshold=0.75, shap_mean_quantile=0.5, shap_sd_quantile=0.5,
    )
    assert result["robustness_class"][0] == "priority"
    assert result["robustness_class"][1] == "background"
