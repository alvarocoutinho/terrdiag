"""
terrdiag.layer2.weights — Spatial weights matrices.

Two kernels implemented from scratch (numpy):
    - gaussian_kernel_weights: W_ij = exp(-d_ij² / (2σ²)), with diagonal=0
    - knn_weights: W_ij = 1 if j is among k nearest neighbors of i, else 0

Both functions accept row_normalize=True (default) so the resulting matrix
has rows summing to 1 (or 0 if no neighbors), which is the convention for
Moran's I and LISA.

Coordinate convention: X, Y in projected CRS (default EPSG:5880, units in
meters). Bandwidth specified in km — conversion handled internally.
"""

from __future__ import annotations

import numpy as np


def _coords_array(X, Y) -> np.ndarray:
    """Stack X, Y into an (n, 2) coordinate array, dropping NaN rows."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.shape != Y.shape:
        raise ValueError(f"X and Y must have same shape; got {X.shape} vs {Y.shape}")
    return np.column_stack([X, Y])


def gaussian_kernel_weights(
    X,
    Y,
    bandwidth_km: float,
    crs_unit: str = "m",
    row_normalize: bool = True,
) -> np.ndarray:
    """Gaussian kernel spatial weights matrix.

    Parameters
    ----------
    X, Y : array-like of length n
        Projected coordinates (one per entity).
    bandwidth_km : float
        Bandwidth σ in kilometers. Weight at distance d is exp(-d²/(2σ²)).
    crs_unit : {'m', 'km'}
        Units of (X, Y). EPSG:5880 (default for terrdiag) is in meters.
    row_normalize : bool
        If True (default), normalize rows to sum to 1. Required for Moran's I
        formula with standard scaling.

    Returns
    -------
    np.ndarray
        (n, n) matrix. Diagonal is zero.
    """
    if bandwidth_km <= 0:
        raise ValueError(f"bandwidth_km must be positive, got {bandwidth_km}")

    coords = _coords_array(X, Y)
    n = len(coords)

    # Pairwise distance — explicit vectorization avoids scipy dep
    diff = coords[:, None, :] - coords[None, :, :]
    d = np.sqrt((diff ** 2).sum(axis=2))

    # Convert to km if needed
    if crs_unit == "m":
        d_km = d / 1000.0
    elif crs_unit == "km":
        d_km = d
    else:
        raise ValueError(f"crs_unit must be 'm' or 'km', got {crs_unit!r}")

    W = np.exp(-0.5 * (d_km / bandwidth_km) ** 2)
    np.fill_diagonal(W, 0.0)

    if row_normalize:
        row_sums = W.sum(axis=1, keepdims=True)
        # Avoid div by zero (isolated points)
        with np.errstate(divide="ignore", invalid="ignore"):
            W = np.where(row_sums > 0, W / row_sums, 0.0)

    return W


def knn_weights(
    X,
    Y,
    k: int,
    row_normalize: bool = True,
) -> np.ndarray:
    """K-nearest-neighbors binary weights matrix.

    Parameters
    ----------
    X, Y : array-like of length n
    k : int
        Number of neighbors per point (excludes self).
    row_normalize : bool
        If True, each row sums to 1 (each neighbor weighted 1/k).

    Returns
    -------
    np.ndarray
        (n, n) matrix.
    """
    if k < 1:
        raise ValueError(f"k must be ≥ 1, got {k}")

    coords = _coords_array(X, Y)
    n = len(coords)
    if k >= n:
        raise ValueError(f"k={k} must be < n={n}")

    diff = coords[:, None, :] - coords[None, :, :]
    d = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(d, np.inf)  # exclude self

    # Indices of k nearest neighbors per row
    nn_idx = np.argpartition(d, k, axis=1)[:, :k]

    W = np.zeros((n, n))
    rows = np.repeat(np.arange(n), k)
    cols = nn_idx.ravel()
    W[rows, cols] = 1.0

    if row_normalize:
        W = W / k

    return W


def inverse_distance_weights(
    X,
    Y,
    cutoff_km: float = 150.0,
    power: float = 1.0,
    crs_unit: str = "m",
    row_normalize: bool = True,
) -> np.ndarray:
    """Inverse distance spatial weights matrix (Chagas et al. 2011 specification).

    Replicates the weight specification used by Chagas, Toneto-Jr & Azzoni
    (2011) in their spatial PSM analysis of Brazilian sugarcane municipalities.

        W_ij = (1/d_ij^power)  if d_ij <= cutoff_km, else 0

    The cutoff is essential to avoid numerical issues with very small weights
    at long distances (Chagas et al. 2011, footnote 11).

    Parameters
    ----------
    X, Y : array-like of length n
    cutoff_km : float
        Maximum distance for non-zero weight. Default 150 km (Chagas main
        specification; robustness checks used 100 and 200 km).
    power : float
        Distance decay exponent. Default 1 (linear inverse).
    crs_unit : {'m', 'km'}
    row_normalize : bool

    Returns
    -------
    np.ndarray
        (n, n) matrix. Diagonal is zero.

    References
    ----------
    Chagas, A.L.S., Toneto-Jr, R., & Azzoni, C.R. (2011). A spatial propensity
    score matching evaluation of the social impacts of sugarcane growing on
    municipalities in Brazil. International Regional Science Review, 35(1),
    48-69.
    """
    if cutoff_km <= 0:
        raise ValueError(f"cutoff_km must be positive, got {cutoff_km}")
    if power <= 0:
        raise ValueError(f"power must be positive, got {power}")

    coords = _coords_array(X, Y)
    n = len(coords)

    diff = coords[:, None, :] - coords[None, :, :]
    d = np.sqrt((diff ** 2).sum(axis=2))
    if crs_unit == "m":
        d_km = d / 1000.0
    elif crs_unit == "km":
        d_km = d
    else:
        raise ValueError(f"crs_unit must be 'm' or 'km', got {crs_unit!r}")

    # Inverse distance with cutoff
    with np.errstate(divide="ignore", invalid="ignore"):
        W = np.where(
            (d_km > 0) & (d_km <= cutoff_km),
            1.0 / (d_km ** power),
            0.0,
        )
    np.fill_diagonal(W, 0.0)

    if row_normalize:
        row_sums = W.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            W = np.where(row_sums > 0, W / row_sums, 0.0)

    return W


def n_neighbors_per_row(W: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """For each row, count how many entries are above threshold (effective n)."""
    return (W > threshold).sum(axis=1)
