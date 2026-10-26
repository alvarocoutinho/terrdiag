"""
terrdiag.layer2.moran — Global Moran's I.

Implements Moran's I with permutation-based inference.

I = (n/S0) · Σᵢ Σⱼ wᵢⱼ (yᵢ − ȳ)(yⱼ − ȳ) / Σᵢ (yᵢ − ȳ)²

For row-normalized W, S0 = n, and I = Σᵢ zᵢ · (Wz)ᵢ / Σᵢ zᵢ²
where z is the deviation from the mean.

Inference: shuffle the attribute vector n_permutations times, compute I on each
shuffle, and report the share of permuted I values that exceed the observed I.
This is the standard one-sided test against the null of spatial randomness.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from terrdiag.audit.hashing import hash_dataframe
from terrdiag.audit.logging import run_log


# ─── Result container ──────────────────────────────────────────────────────


@dataclass(slots=True)
class MoranResult:
    """Global Moran's I result with inference and methods for inspection."""

    I: float
    E_I: float
    p_value: float
    z_score: float
    n: int
    n_permutations: int
    attribute_name: str
    attribute_values: np.ndarray            # standardized z
    spatial_lag: np.ndarray                 # W @ z
    permutation_distribution: np.ndarray
    bandwidth_km: float
    interpretation: str

    def summary(self) -> str:
        return (
            f"Moran's I({self.attribute_name}, bw={self.bandwidth_km}km) = "
            f"{self.I:+.4f} (E[I]={self.E_I:+.4f}, z={self.z_score:+.2f}, "
            f"p={self.p_value:.4f}, n={self.n}) — {self.interpretation}"
        )

    def to_dict(self) -> dict:
        return {
            "attribute_name": self.attribute_name,
            "I": float(self.I),
            "E_I": float(self.E_I),
            "z_score": float(self.z_score),
            "p_value": float(self.p_value),
            "n": int(self.n),
            "n_permutations": int(self.n_permutations),
            "bandwidth_km": float(self.bandwidth_km),
            "interpretation": self.interpretation,
        }

    def plot(self, **kwargs):
        """Moran scatterplot: z(y) on X-axis, spatial lag on Y-axis."""
        from terrdiag.layer2.plotting import plot_moran_scatter
        return plot_moran_scatter(self, **kwargs)


# ─── Core ──────────────────────────────────────────────────────────────────


def _interpret(I: float, E_I: float, p: float, alpha: float = 0.05) -> str:
    if p >= alpha:
        return f"NOT SIGNIFICANT at α={alpha}"
    if I > E_I:
        return f"POSITIVE AUTOCORRELATION at α={alpha} (clustering)"
    return f"NEGATIVE AUTOCORRELATION at α={alpha} (dispersion)"


def moran_i(
    values: np.ndarray | pd.Series,
    weights: np.ndarray,
    n_permutations: int = 999,
    random_state: int = 42,
    bandwidth_km: float = 0.0,
    attribute_name: str = "value",
    alpha: float = 0.05,
    audit: bool = True,
) -> MoranResult:
    """Compute Global Moran's I with permutation inference.

    Parameters
    ----------
    values : array-like, shape (n,)
        Attribute values. NaN values must be removed before calling (caller's
        responsibility; this function asserts no NaN).
    weights : np.ndarray, shape (n, n)
        Spatial weights matrix (typically row-normalized).
    n_permutations : int
        Number of label shuffles for inference. Default 999.
    random_state : int
    bandwidth_km : float
        Only used in result metadata (does not affect computation).
    attribute_name : str
        Stored in the result for plotting/reporting.
    alpha : float
        Significance threshold for interpretation field.
    audit : bool

    Returns
    -------
    MoranResult
    """
    t0 = time.time()
    y = np.asarray(values, dtype=float).ravel()
    if np.isnan(y).any():
        raise ValueError(
            "moran_i: values contain NaN. Filter or impute before passing."
        )
    n = len(y)
    if weights.shape != (n, n):
        raise ValueError(
            f"weights shape {weights.shape} != ({n}, {n}) implied by len(values)"
        )

    z = y - y.mean()
    Wz = weights @ z
    denom = (z ** 2).sum()
    if denom == 0:
        raise ValueError("All values are identical; Moran's I undefined.")

    # For row-normalized W, S0 = sum(W) = n (or whatever the rows sum to).
    S0 = weights.sum()
    I_obs = (n / S0) * (z * Wz).sum() / denom

    # Permutation inference: shuffle y, recompute I
    rng = np.random.default_rng(random_state)
    perm_I = np.empty(n_permutations, dtype=float)
    idx = np.arange(n)
    for i in range(n_permutations):
        rng.shuffle(idx)
        z_perm = y[idx] - y.mean()
        perm_I[i] = (n / S0) * (z_perm * (weights @ z_perm)).sum() / (z_perm ** 2).sum()

    # Two-sided p: share of permuted values as extreme as observed
    if I_obs >= 0:
        p_value = float((perm_I >= I_obs).sum() + 1) / (n_permutations + 1)
    else:
        p_value = float((perm_I <= I_obs).sum() + 1) / (n_permutations + 1)

    E_I = -1.0 / (n - 1)
    perm_var = float(perm_I.var())
    z_score = (I_obs - E_I) / np.sqrt(perm_var) if perm_var > 0 else 0.0

    # Standardized z for scatterplot
    z_std = z / z.std(ddof=0)
    Wz_std = weights @ z_std

    interp = _interpret(I_obs, E_I, p_value, alpha=alpha)

    result = MoranResult(
        I=float(I_obs),
        E_I=float(E_I),
        p_value=float(p_value),
        z_score=float(z_score),
        n=n,
        n_permutations=n_permutations,
        attribute_name=attribute_name,
        attribute_values=z_std,
        spatial_lag=Wz_std,
        permutation_distribution=perm_I,
        bandwidth_km=bandwidth_km,
        interpretation=interp,
    )

    duration = time.time() - t0
    if audit:
        run_log.record(
            step="layer2.moran_i",
            duration_s=duration,
            params={
                "attribute_name": attribute_name,
                "bandwidth_km": bandwidth_km,
                "n_permutations": n_permutations,
                "alpha": alpha,
            },
            input_rows=n,
            output_rows=1,
            output_hash=None,
            notes=result.summary(),
        )

    return result
