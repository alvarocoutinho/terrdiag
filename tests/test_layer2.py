"""
Tests for terrdiag.layer2 — weights, Moran's I, LISA, sensitivity.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from terrdiag import layer2


# ─── Fixtures ───────────────────────────────────────────────────────────


def _grid_points(side: int = 10):
    """Regular grid of points in a square."""
    xs = np.linspace(0, 1e6, side)
    ys = np.linspace(0, 1e6, side)
    X, Y = np.meshgrid(xs, ys)
    return X.ravel(), Y.ravel()


def _random_attribute(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=n)


# ─── Weights ────────────────────────────────────────────────────────────


def test_gaussian_kernel_basic():
    X, Y = _grid_points(side=5)
    W = layer2.gaussian_kernel_weights(X, Y, bandwidth_km=200)
    assert W.shape == (25, 25)
    assert (np.diag(W) == 0).all()
    # Row-normalized
    np.testing.assert_allclose(W.sum(axis=1), 1.0, atol=1e-10)


def test_gaussian_kernel_units():
    X, Y = _grid_points(side=5)
    # Same kernel applied with meters vs already-km coords should differ
    W_m = layer2.gaussian_kernel_weights(X, Y, bandwidth_km=100, crs_unit="m")
    W_km = layer2.gaussian_kernel_weights(X / 1000, Y / 1000, bandwidth_km=100, crs_unit="km")
    np.testing.assert_allclose(W_m, W_km, atol=1e-10)


def test_knn_weights():
    X, Y = _grid_points(side=5)
    W = layer2.knn_weights(X, Y, k=4)
    assert W.shape == (25, 25)
    assert (np.diag(W) == 0).all()
    # Each row has exactly 4 nonzero with row-normalize → sum 1
    np.testing.assert_allclose(W.sum(axis=1), 1.0, atol=1e-10)
    assert ((W > 0).sum(axis=1) == 4).all()


def test_n_neighbors_per_row():
    X, Y = _grid_points(side=5)
    W = layer2.knn_weights(X, Y, k=3)
    n_nb = layer2.n_neighbors_per_row(W)
    assert (n_nb == 3).all()


# ─── Moran ──────────────────────────────────────────────────────────────


def test_moran_random_is_zero():
    """Random attribute should give I ≈ E[I] = -1/(n-1)."""
    n_side = 8
    X, Y = _grid_points(side=n_side)
    y = _random_attribute(n_side * n_side, seed=123)
    W = layer2.gaussian_kernel_weights(X, Y, bandwidth_km=300)
    m = layer2.moran_i(y, W, n_permutations=99, attribute_name="random")
    # E[I] = -1/(n-1) ≈ -0.016 for n=64
    assert abs(m.I) < 0.2, f"Random gave I={m.I:.3f}"
    assert m.p_value > 0.05, "Random should not be significant"


def test_moran_gradient_is_positive():
    """Gradient attribute should yield strong positive Moran."""
    n_side = 8
    X, Y = _grid_points(side=n_side)
    grid = np.arange(n_side).reshape(-1, 1).repeat(n_side, axis=1).T
    y = grid.ravel().astype(float)
    W = layer2.gaussian_kernel_weights(X, Y, bandwidth_km=200)
    m = layer2.moran_i(y, W, n_permutations=99, attribute_name="gradient")
    assert m.I > 0.3, f"Gradient gave I={m.I:.3f}, expected > 0.3"
    assert m.p_value < 0.05


def test_moran_handles_nan_input():
    n_side = 5
    X, Y = _grid_points(side=n_side)
    y = _random_attribute(n_side * n_side)
    y[0] = np.nan
    W = layer2.gaussian_kernel_weights(X, Y, bandwidth_km=300)
    with pytest.raises(ValueError, match="NaN"):
        layer2.moran_i(y, W, n_permutations=10)


def test_moran_result_methods():
    n_side = 6
    X, Y = _grid_points(side=n_side)
    y = _random_attribute(n_side * n_side)
    W = layer2.gaussian_kernel_weights(X, Y, bandwidth_km=300)
    m = layer2.moran_i(y, W, n_permutations=99)
    assert isinstance(m.summary(), str)
    d = m.to_dict()
    assert "I" in d
    fig = m.plot()
    assert fig is not None
    plt.close(fig)


# ─── LISA ───────────────────────────────────────────────────────────────


def test_lisa_basic():
    n_side = 8
    X, Y = _grid_points(side=n_side)
    grid = np.arange(n_side).reshape(-1, 1).repeat(n_side, axis=1).T
    y = grid.ravel().astype(float)
    W = layer2.gaussian_kernel_weights(X, Y, bandwidth_km=200)
    lisa = layer2.local_moran(y, W, n_permutations=99, attribute_name="gradient", bandwidth_km=200)
    # With a strong gradient, both extremes should produce HH and LL points
    counts = lisa.classification_counts()
    n = n_side * n_side
    assert len(lisa.local_I) == n
    # At least some significant points
    assert (lisa.p_values < 0.05).sum() > 0


def test_lisa_to_dataframe():
    n_side = 5
    X, Y = _grid_points(side=n_side)
    y = _random_attribute(n_side * n_side)
    W = layer2.gaussian_kernel_weights(X, Y, bandwidth_km=300)
    lisa = layer2.local_moran(y, W, n_permutations=99)
    df = lisa.to_dataframe()
    assert "local_I" in df.columns
    assert "classification" in df.columns
    assert len(df) == n_side * n_side


def test_lisa_classify_alpha():
    n_side = 6
    X, Y = _grid_points(side=n_side)
    y = _random_attribute(n_side * n_side, seed=2)
    W = layer2.gaussian_kernel_weights(X, Y, bandwidth_km=300)
    lisa = layer2.local_moran(y, W, n_permutations=99)
    # Re-classify at stricter alpha — fewer significant
    cls_strict = lisa.classify(alpha=0.01)
    cls_loose = lisa.classify(alpha=0.10)
    n_strict = (cls_strict != "ns").sum()
    n_loose = (cls_loose != "ns").sum()
    assert n_strict <= n_loose


def test_lisa_fdr_correction():
    n_side = 6
    X, Y = _grid_points(side=n_side)
    y = _random_attribute(n_side * n_side, seed=3)
    W = layer2.gaussian_kernel_weights(X, Y, bandwidth_km=300)
    lisa = layer2.local_moran(y, W, n_permutations=99)
    # FDR should give >= raw p-values
    assert (lisa.p_values_fdr >= lisa.p_values - 1e-10).all()


def test_lisa_plot_kinds():
    n_side = 5
    X, Y = _grid_points(side=n_side)
    y = _random_attribute(n_side * n_side)
    W = layer2.gaussian_kernel_weights(X, Y, bandwidth_km=300)
    lisa = layer2.local_moran(y, W, n_permutations=99, bandwidth_km=300)
    panel = pd.DataFrame({"X": X, "Y": Y})
    for kind in ["scatter", "counts"]:
        fig = lisa.plot(kind=kind)
        assert fig is not None
        plt.close(fig)
    fig = lisa.plot(kind="map", panel=panel)
    assert fig is not None
    plt.close(fig)


def test_lisa_invalid_kind():
    n_side = 4
    X, Y = _grid_points(side=n_side)
    y = _random_attribute(n_side * n_side)
    W = layer2.gaussian_kernel_weights(X, Y, bandwidth_km=300)
    lisa = layer2.local_moran(y, W, n_permutations=20)
    with pytest.raises(ValueError, match="Unknown plot kind"):
        lisa.plot(kind="nonexistent")


# ─── Sensitivity ────────────────────────────────────────────────────────


def test_bandwidth_sweep():
    n_side = 6
    X, Y = _grid_points(side=n_side)
    y = _random_attribute(n_side * n_side)
    df = layer2.bandwidth_sweep(
        y, X, Y, bandwidths_km=[50, 100, 200], n_permutations=49,
    )
    assert len(df) == 3
    assert "I" in df.columns
    assert "p_value" in df.columns


def test_sensitivity_plot():
    n_side = 5
    X, Y = _grid_points(side=n_side)
    y = _random_attribute(n_side * n_side)
    df = layer2.bandwidth_sweep(
        y, X, Y, bandwidths_km=[50, 100, 200], n_permutations=29,
    )
    fig = layer2.plot_sensitivity(df, attribute_name="test")
    assert fig is not None
    plt.close(fig)


# ─── Inspect ────────────────────────────────────────────────────────────


def test_lisa_classification_counts():
    n_side = 5
    X, Y = _grid_points(side=n_side)
    y = _random_attribute(n_side * n_side)
    W = layer2.gaussian_kernel_weights(X, Y, bandwidth_km=300)
    lisa = layer2.local_moran(y, W, n_permutations=49)
    counts = layer2.lisa_classification_counts(lisa)
    assert "count" in counts.columns
    assert "share" in counts.columns
    assert set(counts.index) == {"HH", "LL", "HL", "LH", "ns"}


def test_lisa_top_contributors():
    n_side = 5
    X, Y = _grid_points(side=n_side)
    y = _random_attribute(n_side * n_side)
    W = layer2.gaussian_kernel_weights(X, Y, bandwidth_km=300)
    lisa = layer2.local_moran(y, W, n_permutations=49)
    panel = pd.DataFrame({"X": X, "Y": Y, "id": range(len(X))})
    top = layer2.lisa_top_contributors(lisa, panel, n=5)
    assert len(top) == 5


def test_significance_diagnostics():
    n_side = 5
    X, Y = _grid_points(side=n_side)
    y = _random_attribute(n_side * n_side)
    W = layer2.gaussian_kernel_weights(X, Y, bandwidth_km=300)
    lisa = layer2.local_moran(y, W, n_permutations=49)
    diag = layer2.lisa_significance_diagnostics(lisa)
    assert "alpha" in diag.columns
    assert "n_significant_raw" in diag.columns
    assert "n_significant_fdr" in diag.columns
