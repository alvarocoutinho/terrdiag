"""
Smoke tests for terrdiag.layer1.inspect and .plotting.

The plotting functions are tested only that they return a matplotlib Figure
without exceptions — visual correctness is checked manually.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for CI

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from terrdiag import layer1
from terrdiag.specs import FeatureRecipe, FeatureSpec


def _build_clustered_panel(n: int = 60, seed: int = 0) -> tuple[pd.DataFrame, "layer1.ClusterResult"]:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "cd_ibge": [f"{i:07d}" for i in range(n)],
            "uf": rng.choice(["SP", "GO", "MG"], size=n),
            "X": rng.uniform(0, 1e6, n),
            "Y": rng.uniform(0, 1e6, n),
            "f1": rng.normal(0, 1, n),
            "f2": rng.normal(0, 1, n),
            "f3": rng.normal(0, 1, n),
            "f4": rng.normal(0, 1, n),
            "delta_neea": rng.normal(2, 5, n),
        }
    )
    features = tuple(
        FeatureSpec(name=c, source_column=c) for c in ["f1", "f2", "f3", "f4"]
    )
    recipe = FeatureRecipe(
        name="r", features=features, pca_components=2, k_clusters=3, n_bootstrap=10,
    )
    result = layer1.cluster(df, recipe)
    return df, result


# ─── Inspect functions ──────────────────────────────────────────────────


def test_cluster_profile():
    df, result = _build_clustered_panel()
    prof = layer1.cluster_profile(result, df)
    assert set(result.feature_names).issubset(prof.index)
    assert "overall" in prof.columns


def test_cluster_profile_both():
    df, result = _build_clustered_panel()
    prof = layer1.cluster_profile(result, df, statistic="both")
    assert isinstance(prof.columns, pd.MultiIndex)


def test_feature_loadings():
    df, result = _build_clustered_panel()
    L = layer1.feature_loadings(result)
    assert L.shape == (len(result.feature_names), 2)
    assert list(L.columns) == ["PC1", "PC2"]


def test_cluster_crosstab():
    df, result = _build_clustered_panel()
    ct = layer1.cluster_crosstab(result, df, group_column="uf")
    assert ct.values.sum() == len(df)


def test_cluster_crosstab_normalized():
    df, result = _build_clustered_panel()
    ct = layer1.cluster_crosstab(result, df, group_column="uf", normalize=True)
    np.testing.assert_allclose(ct.sum(axis=1).values, 1.0)


def test_centroid_distances():
    df, result = _build_clustered_panel()
    d = layer1.centroid_distances(result)
    assert d.shape == (3, 3)
    np.testing.assert_allclose(np.diag(d), 0.0, atol=1e-9)
    # symmetry
    np.testing.assert_allclose(d.values, d.values.T)


def test_naming_suggestion():
    df, result = _build_clustered_panel()
    # Add the columns naming_suggestion looks for
    df["log_pib_pc"] = df["f1"]
    df["natveg_share"] = df["f2"]
    df["log_trator_per_est"] = df["f3"]
    df["share_est_at"] = df["f4"]
    # Recluster on these explicit names
    features = tuple(
        FeatureSpec(name=c, source_column=c)
        for c in ["log_pib_pc", "natveg_share", "log_trator_per_est", "share_est_at"]
    )
    recipe = FeatureRecipe(
        name="r", features=features, pca_components=2, k_clusters=3, n_bootstrap=5,
    )
    result = layer1.cluster(df, recipe)
    suggest = layer1.naming_suggestion(result, df)
    assert len(suggest) == 3
    assert "suggested_name" in suggest.columns
    assert all(suggest["suggested_name"].str.len() > 0)


# ─── ClusterResult helper methods ────────────────────────────────────────


def test_result_methods():
    df, result = _build_clustered_panel()
    assert isinstance(result.summary(), str)
    assert isinstance(result.profile(df), pd.DataFrame)
    assert isinstance(result.loadings(), pd.DataFrame)
    rep = result.report(df)
    assert "summary" in rep
    assert "cluster_sizes" in rep
    assert "ari_bootstrap_ci95" in rep


# ─── Plotting smoke tests ────────────────────────────────────────────────


@pytest.mark.parametrize("kind", [
    "pca", "centroids", "loadings", "profile", "ari", "silhouette"
])
def test_plot_dispatch(kind):
    df, result = _build_clustered_panel()
    fig = result.plot(kind=kind, **({"panel": df} if kind == "profile" else {}))
    assert fig is not None
    plt.close(fig)


def test_plot_by_uf():
    df, result = _build_clustered_panel()
    fig = layer1.plot_cluster_by_group(result, df, group_column="uf")
    assert fig is not None
    plt.close(fig)


def test_plot_xy_map():
    df, result = _build_clustered_panel()
    fig = layer1.plot_xy_map(result, df)
    assert fig is not None
    plt.close(fig)


def test_plot_outcome_by_cluster():
    df, result = _build_clustered_panel()
    test = layer1.kruskal_wallis(df, result.labels, "delta_neea")
    fig = layer1.plot_outcome_by_cluster(test, df, result.labels)
    assert fig is not None
    plt.close(fig)


def test_plot_invalid_kind():
    df, result = _build_clustered_panel()
    with pytest.raises(ValueError, match="Unknown plot kind"):
        result.plot(kind="nonexistent")


# ─── IndependenceTest helpers ────────────────────────────────────────────


def test_independence_test_helpers():
    df, result = _build_clustered_panel()
    test = layer1.kruskal_wallis(df, result.labels, "delta_neea")
    assert isinstance(test.summary(), str)
    d = test.to_dict()
    assert "outcome_name" in d
    assert "p_value" in d
    fig = test.plot(df, result.labels)
    assert fig is not None
    plt.close(fig)
