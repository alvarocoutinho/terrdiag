"""
Tests for terrdiag.layer1 — preprocess, impute, cluster, independence.

Includes the NEEA GUARD-RAIL regression test, which must never break.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from terrdiag import layer1
from terrdiag.specs import FeatureRecipe, FeatureSpec


# ─── Fixtures ────────────────────────────────────────────────────────────


def _synthetic_muni(n: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "cd_ibge": [f"{i:07d}" for i in range(n)],
            "uf": rng.choice(["SP", "GO", "MG"], size=n),
            "n_usinas": rng.integers(1, 5, size=n),
            "feat_a": rng.normal(0, 1, n),
            "feat_b": rng.normal(0, 1, n),
            "feat_c": rng.normal(0, 1, n),
            "feat_d": rng.normal(0, 1, n),
            "feat_e": rng.normal(0, 1, n),
            "feat_share": rng.uniform(0, 1, n),
            # outcome — never in features
            "delta_neea": rng.normal(2, 5, n),
        }
    )


def _basic_features() -> tuple[FeatureSpec, ...]:
    return tuple(
        FeatureSpec(name=c, source_column=c)
        for c in ["feat_a", "feat_b", "feat_c", "feat_d", "feat_e", "feat_share"]
    )


# ─── preprocess ──────────────────────────────────────────────────────────


def test_preprocess_identity_is_noop():
    df = _synthetic_muni()
    specs = _basic_features()
    out = layer1.preprocess(df, specs)
    pd.testing.assert_frame_equal(df.reset_index(drop=True), out.reset_index(drop=True))


def test_preprocess_log1p_and_clip():
    df = pd.DataFrame({"x": [0.0, 10.0, 100.0, -5.0]})
    spec = FeatureSpec(name="x", source_column="x", transform="log1p")
    out = layer1.preprocess(df, [spec])
    # log1p of negatives clipped to 0 → log1p(0) = 0
    assert out["x"].iloc[0] == 0.0
    assert out["x"].iloc[3] == 0.0
    assert out["x"].iloc[1] == pytest.approx(np.log1p(10.0))


def test_preprocess_clip():
    df = pd.DataFrame({"s": [-0.5, 0.3, 1.5]})
    spec = FeatureSpec(name="s", source_column="s", clip=(0.0, 1.0))
    out = layer1.preprocess(df, [spec])
    assert list(out["s"]) == [0.0, 0.3, 1.0]


def test_preprocess_zscore():
    df = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
    out = layer1.preprocess(df, [FeatureSpec(name="x", source_column="x", transform="zscore")])
    assert out["x"].mean() == pytest.approx(0.0, abs=1e-9)


# ─── impute ──────────────────────────────────────────────────────────────


def test_impute_median_by_group():
    df = pd.DataFrame({"uf": ["SP", "SP", "GO", "GO"], "x": [1.0, np.nan, 5.0, 7.0]})
    out = layer1.impute(df, ["x"], strategy="median_by_group", group_column="uf")
    # Median of SP (only [1.0]) is 1.0
    assert out["x"].iloc[1] == 1.0


def test_impute_global_fallback():
    """If a group has all NaN, fall back to global median."""
    df = pd.DataFrame({"uf": ["SP", "SP", "GO"], "x": [np.nan, np.nan, 4.0]})
    out = layer1.impute(df, ["x"], strategy="median_by_group", group_column="uf")
    # SP group is all-NaN, falls back to global median = 4.0
    assert out["x"].iloc[0] == 4.0


def test_impute_drop():
    df = pd.DataFrame({"uf": ["SP", "SP", "GO"], "x": [1.0, np.nan, 3.0]})
    out = layer1.impute(df, ["x"], strategy="drop")
    assert len(out) == 2


# ═══ THE GUARD-RAIL ═══════════════════════════════════════════════════════


class TestNEEAGuardRail:
    """NEEA must never reach Layer 1. These tests are a methodological invariant."""

    def test_rejects_feature_named_neea(self):
        bad = FeatureSpec(name="neea_t0_2022", source_column="x")
        recipe = FeatureRecipe(
            name="bad", features=(bad,) + _basic_features(),
            pca_components=2, k_clusters=2,
        )
        with pytest.raises(ValueError, match="GUARD-RAIL"):
            layer1.cluster(_synthetic_muni(), recipe)

    def test_rejects_delta_neea(self):
        bad = FeatureSpec(name="delta_neea", source_column="delta_neea")
        recipe = FeatureRecipe(
            name="bad", features=(bad,) + _basic_features(),
            pca_components=2, k_clusters=2,
        )
        with pytest.raises(ValueError, match="GUARD-RAIL"):
            layer1.cluster(_synthetic_muni(), recipe)

    def test_rejects_vol_pct(self):
        bad = FeatureSpec(name="vol_pct_first", source_column="vol_pct_first")
        recipe = FeatureRecipe(
            name="bad", features=(bad,) + _basic_features(),
            pca_components=2, k_clusters=2,
        )
        with pytest.raises(ValueError, match="GUARD-RAIL"):
            layer1.cluster(_synthetic_muni(), recipe)

    def test_rejects_cbio_and_vol_eleg(self):
        for nm in ["cbio_share", "vol_eleg_pct", "VOL_ELEG", "neea_diff", "CBIO_X"]:
            bad = FeatureSpec(name=nm, source_column=nm)
            recipe = FeatureRecipe(
                name="bad", features=(bad,) + _basic_features(),
                pca_components=2, k_clusters=2,
            )
            with pytest.raises(ValueError, match="GUARD-RAIL"):
                layer1.cluster(_synthetic_muni(), recipe)

    def test_rejects_source_column_neea_even_if_name_clean(self):
        """Even if the FeatureSpec.name is innocent, the source_column matters."""
        bad = FeatureSpec(name="innocent_name", source_column="neea_t0_2022")
        recipe = FeatureRecipe(
            name="bad", features=(bad,) + _basic_features(),
            pca_components=2, k_clusters=2,
        )
        with pytest.raises(ValueError, match="GUARD-RAIL"):
            layer1.cluster(_synthetic_muni(), recipe)

    def test_test_override_works_only_explicitly(self):
        """_allow_outcome_features bypasses the guard-rail (for testing only)."""
        # name triggers guard-rail; source_column exists in synthetic data
        bad = FeatureSpec(name="neea_metric", source_column="delta_neea")
        recipe = FeatureRecipe(
            name="bad_but_ok", features=(bad,) + _basic_features(),
            pca_components=2, k_clusters=2, n_bootstrap=5,
        )
        # Should not raise when override is set
        result = layer1.cluster(_synthetic_muni(), recipe, _allow_outcome_features=True)
        assert result.labels is not None


# ─── cluster — basic correctness ─────────────────────────────────────────


def test_cluster_runs_and_returns_result():
    df = _synthetic_muni(n=60)
    recipe = FeatureRecipe(
        name="basic", features=_basic_features(),
        pca_components=2, k_clusters=3, n_bootstrap=20,
    )
    res = layer1.cluster(df, recipe)
    assert res.labels.shape == (60,)
    assert set(res.labels) <= {0, 1, 2}
    assert res.pca_components.shape == (60, 2)
    assert res.centroids.shape == (3, 2)
    assert res.ari_bootstrap.shape == (20,)
    assert len(res.feature_names) == 6


def test_cluster_drops_high_missingness():
    df = _synthetic_muni(n=60)
    df.loc[df.index[:50], "feat_a"] = np.nan  # 83% missing
    specs = (FeatureSpec(name="feat_a", source_column="feat_a", missing_threshold=0.20),) + tuple(
        FeatureSpec(name=c, source_column=c) for c in ["feat_b", "feat_c", "feat_d", "feat_e"]
    )
    recipe = FeatureRecipe(
        name="drop_test", features=specs, pca_components=2, k_clusters=2, n_bootstrap=10,
    )
    res = layer1.cluster(df, recipe)
    assert "feat_a" not in res.feature_names
    assert res.n_dropped_features == 1


def test_cluster_raises_on_remaining_nan():
    df = _synthetic_muni(n=60)
    df.loc[df.index[0], "feat_a"] = np.nan  # below threshold but present
    recipe = FeatureRecipe(
        name="x", features=_basic_features(), pca_components=2, k_clusters=2, n_bootstrap=5,
    )
    with pytest.raises(ValueError, match="NaN"):
        layer1.cluster(df, recipe)


def test_cluster_reproducible():
    """Same recipe + data → identical labels."""
    df = _synthetic_muni(n=60)
    recipe = FeatureRecipe(
        name="r", features=_basic_features(), pca_components=2, k_clusters=3,
        random_state=42, n_bootstrap=10,
    )
    r1 = layer1.cluster(df, recipe)
    r2 = layer1.cluster(df, recipe)
    np.testing.assert_array_equal(r1.labels, r2.labels)
    np.testing.assert_allclose(r1.ari_bootstrap, r2.ari_bootstrap)


# ─── independence ────────────────────────────────────────────────────────


def test_kruskal_wallis_basic():
    df = _synthetic_muni(n=60)
    labels = np.tile([0, 1, 2], 20)
    test = layer1.kruskal_wallis(df, labels, "delta_neea")
    assert test.outcome_name == "delta_neea"
    assert test.n_clusters == 3
    assert 0.0 < test.p_value < 1.0


def test_kruskal_wallis_missing_column():
    df = _synthetic_muni(n=30)
    labels = np.zeros(30, dtype=int)
    with pytest.raises(KeyError):
        layer1.kruskal_wallis(df, labels, "nonexistent_col")


def test_kruskal_wallis_length_mismatch():
    df = _synthetic_muni(n=30)
    with pytest.raises(ValueError, match="len"):
        layer1.kruskal_wallis(df, np.zeros(29), "delta_neea")


def test_kruskal_wallis_detects_real_difference():
    """If outcome truly differs by cluster, p should be < 0.05."""
    df = pd.DataFrame({"out": np.concatenate([np.zeros(50), np.full(50, 10.0)])})
    labels = np.concatenate([np.zeros(50, dtype=int), np.ones(50, dtype=int)])
    test = layer1.kruskal_wallis(df, labels, "out")
    assert test.p_value < 0.001
