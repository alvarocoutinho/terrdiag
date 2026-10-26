"""
terrdiag.layer2.sensitivity — Bandwidth sensitivity analysis.

For a given attribute, sweep across multiple bandwidths to confirm that the
spatial autocorrelation result is robust to the kernel scale choice.
The output table lets the analyst see how I and its significance change with
bandwidth.
"""

from __future__ import annotations

import time
from typing import Iterable

import numpy as np
import pandas as pd

from terrdiag.audit.logging import run_log
from terrdiag.layer2.moran import moran_i
from terrdiag.layer2.weights import gaussian_kernel_weights


def bandwidth_sweep(
    values: np.ndarray | pd.Series,
    X: np.ndarray | pd.Series,
    Y: np.ndarray | pd.Series,
    bandwidths_km: Iterable[float] = (50.0, 75.0, 100.0, 125.0, 150.0, 200.0),
    n_permutations: int = 499,
    random_state: int = 42,
    attribute_name: str = "value",
    audit: bool = True,
) -> pd.DataFrame:
    """Compute Moran's I across a range of Gaussian kernel bandwidths.

    Parameters
    ----------
    values : array-like, length n
    X, Y : array-like, length n
        Projected coordinates.
    bandwidths_km : iterable of float
        Bandwidths to test.
    n_permutations : int
        Permutations per bandwidth (kept moderate for speed).
    random_state : int
    attribute_name : str

    Returns
    -------
    pd.DataFrame
        Columns: bandwidth_km, I, E_I, z_score, p_value, n_eff_mean,
        interpretation.
    """
    t0 = time.time()
    bandwidths_km = list(bandwidths_km)
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    y = np.asarray(values, dtype=float)

    rows = []
    for bw in bandwidths_km:
        W = gaussian_kernel_weights(X, Y, bandwidth_km=bw)
        m = moran_i(
            y, W,
            n_permutations=n_permutations, random_state=random_state,
            bandwidth_km=bw, attribute_name=attribute_name, audit=False,
        )
        # Mean effective number of neighbors (per row, threshold >1e-6)
        n_eff = float((W > 1e-6).sum(axis=1).mean())
        rows.append({
            "bandwidth_km": float(bw),
            "I": float(m.I),
            "E_I": float(m.E_I),
            "z_score": float(m.z_score),
            "p_value": float(m.p_value),
            "n_eff_mean": n_eff,
            "interpretation": m.interpretation,
        })

    df = pd.DataFrame(rows)

    duration = time.time() - t0
    if audit:
        run_log.record(
            step="layer2.bandwidth_sweep",
            duration_s=duration,
            params={
                "attribute_name": attribute_name,
                "bandwidths_km": bandwidths_km,
                "n_permutations": n_permutations,
            },
            input_rows=len(y),
            output_rows=len(df),
            output_hash=None,
            notes=(
                f"swept {len(df)} bandwidths; "
                f"I range [{df['I'].min():.3f}, {df['I'].max():.3f}]; "
                f"min p={df['p_value'].min():.4f}"
            ),
        )

    return df


def plot_sensitivity(sweep_df: pd.DataFrame, attribute_name: str = ""):
    """Plot I and p-value across bandwidths.

    Returns a matplotlib Figure.
    """
    from terrdiag.layer2.plotting import plot_bandwidth_sensitivity
    return plot_bandwidth_sensitivity(sweep_df, attribute_name=attribute_name)
