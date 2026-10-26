"""
terrdiag.layer2.lisa — Local Indicators of Spatial Association.

For each entity i:
    Iᵢ = zᵢ · Σⱼ wᵢⱼ · zⱼ

Inference: conditional permutation — fix zᵢ, shuffle the other n−1 z values,
recompute the pseudo-Iᵢ, repeat. The pseudo p-value is the share of pseudo
values as extreme as observed.

Classification at significance level α:
    HH — high attribute, high spatial lag, p < α    (hot spot)
    LL — low  attribute, low  spatial lag, p < α    (cold spot)
    HL — high attribute, low  spatial lag, p < α    (high outlier)
    LH — low  attribute, high spatial lag, p < α    (low outlier)
    ns — p ≥ α                                       (not significant)

Caveat: pseudo p-values from permutation are not corrected for multiple
comparisons. For strict inference at α=0.05 with n=230 entities, consider
FDR correction (Benjamini-Hochberg) — exposed via classify(fdr=True).
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
class LISAResult:
    """Local Moran's I result with per-entity classification and inspection."""

    local_I: np.ndarray              # (n,) — local Moran values
    z_attribute: np.ndarray          # (n,) — standardized attribute
    spatial_lag: np.ndarray          # (n,) — W @ z
    p_values: np.ndarray             # (n,) — pseudo p-values from permutation
    p_values_fdr: np.ndarray         # (n,) — Benjamini-Hochberg adjusted p
    classifications: np.ndarray     # (n,) — HH/LL/HL/LH/ns at default alpha
    alpha: float
    n_permutations: int
    bandwidth_km: float
    attribute_name: str
    moran_I_global: float            # global I, for reference

    def summary(self) -> str:
        counts = self.classification_counts()
        return (
            f"LISA({self.attribute_name}, bw={self.bandwidth_km}km, "
            f"α={self.alpha}, perms={self.n_permutations}): "
            f"global I={self.moran_I_global:+.3f}, "
            f"HH={counts.get('HH', 0)}, LL={counts.get('LL', 0)}, "
            f"HL={counts.get('HL', 0)}, LH={counts.get('LH', 0)}, "
            f"ns={counts.get('ns', 0)}, n={len(self.local_I)}"
        )

    def classification_counts(self) -> dict:
        u, c = np.unique(self.classifications, return_counts=True)
        return {str(k): int(v) for k, v in zip(u, c)}

    def classify(self, alpha: float | None = None, use_fdr: bool = False) -> np.ndarray:
        """Re-classify at a different α (or with FDR correction).

        Returns
        -------
        np.ndarray of strings, length n.
        """
        a = alpha if alpha is not None else self.alpha
        p = self.p_values_fdr if use_fdr else self.p_values
        sig = p < a
        cls = np.full(len(self.local_I), "ns", dtype="<U2")
        # 4 quadrants for significant points
        hh = sig & (self.z_attribute > 0) & (self.spatial_lag > 0)
        ll_ = sig & (self.z_attribute < 0) & (self.spatial_lag < 0)
        hl = sig & (self.z_attribute > 0) & (self.spatial_lag < 0)
        lh = sig & (self.z_attribute < 0) & (self.spatial_lag > 0)
        cls[hh] = "HH"
        cls[ll_] = "LL"
        cls[hl] = "HL"
        cls[lh] = "LH"
        return cls

    def to_dataframe(self, panel: pd.DataFrame | None = None) -> pd.DataFrame:
        """Per-entity table: local I, p-value, classification."""
        df = pd.DataFrame(
            {
                "z_attribute": self.z_attribute,
                "spatial_lag": self.spatial_lag,
                "local_I": self.local_I,
                "p_value": self.p_values,
                "p_value_fdr": self.p_values_fdr,
                "classification": self.classifications,
            }
        )
        if panel is not None and len(panel) == len(df):
            df = pd.concat([panel.reset_index(drop=True), df], axis=1)
        return df

    def to_dict(self) -> dict:
        return {
            "attribute_name": self.attribute_name,
            "bandwidth_km": float(self.bandwidth_km),
            "alpha": float(self.alpha),
            "n_permutations": int(self.n_permutations),
            "moran_I_global": float(self.moran_I_global),
            "n": int(len(self.local_I)),
            "counts": self.classification_counts(),
        }

    def plot(self, kind: str = "scatter", **kwargs):
        """Diagnostic plots.

        Available kinds:
          - 'scatter': Moran scatter colored by classification
          - 'map': spatial scatter colored by HH/LL/HL/LH/ns
          - 'counts': bar chart of classification counts
        """
        from terrdiag.layer2 import plotting
        dispatch = {
            "scatter": plotting.plot_lisa_scatter,
            "map": plotting.plot_lisa_map,
            "counts": plotting.plot_lisa_counts,
        }
        if kind not in dispatch:
            raise ValueError(
                f"Unknown plot kind: {kind!r}. Available: {sorted(dispatch.keys())}"
            )
        return dispatch[kind](self, **kwargs)


# ─── BH FDR adjustment ────────────────────────────────────────────────────


def _bh_fdr(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * n / np.arange(1, n + 1)
    # Enforce monotone non-increasing in reverse
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out = np.empty_like(adj)
    out[order] = adj
    return out


# ─── Core ──────────────────────────────────────────────────────────────────


def local_moran(
    values: np.ndarray | pd.Series,
    weights: np.ndarray,
    n_permutations: int = 999,
    random_state: int = 42,
    alpha: float = 0.05,
    bandwidth_km: float = 0.0,
    attribute_name: str = "value",
    audit: bool = True,
) -> LISAResult:
    """Compute Local Moran's I for each entity with conditional permutation.

    Parameters
    ----------
    values : array-like of length n
    weights : (n, n) np.ndarray
        Spatial weights matrix. Typically row-normalized.
    n_permutations : int
    random_state : int
    alpha : float
        Significance threshold for default classification.
    bandwidth_km : float
        Metadata only.
    attribute_name : str
    audit : bool

    Returns
    -------
    LISAResult
    """
    t0 = time.time()
    y = np.asarray(values, dtype=float).ravel()
    if np.isnan(y).any():
        raise ValueError("local_moran: values contain NaN")
    n = len(y)
    if weights.shape != (n, n):
        raise ValueError(f"weights shape {weights.shape} != ({n}, {n})")

    # Standardize
    z = (y - y.mean()) / y.std(ddof=0)
    Wz = weights @ z
    local_I = z * Wz

    # Conditional permutation per point
    rng = np.random.default_rng(random_state)
    p_values = np.empty(n, dtype=float)

    # Pre-allocate index array for shuffling
    base_idx = np.arange(n)

    for i in range(n):
        # All indices except i
        others_mask = base_idx != i
        z_others = z[others_mask]
        w_row = weights[i, others_mask]  # neighbor weights (excluding self)

        # Permute neighbors of i: shuffle z values among the n-1 other points,
        # compute pseudo Wz_i = w_row @ z_shuffled[k_first_n_others]
        # Simpler: shuffle z_others and dot with w_row
        # Vectorize across permutations
        # For each perm: pseudo_I_i = z[i] * (w_row @ z_others_perm)
        perm_I_i = np.empty(n_permutations, dtype=float)
        for k in range(n_permutations):
            perm = rng.permutation(len(z_others))
            perm_I_i[k] = z[i] * (w_row * z_others[perm]).sum()

        obs = local_I[i]
        if obs >= 0:
            p_values[i] = (((perm_I_i >= obs).sum() + 1) / (n_permutations + 1))
        else:
            p_values[i] = (((perm_I_i <= obs).sum() + 1) / (n_permutations + 1))

    p_fdr = _bh_fdr(p_values)

    # Classify at default alpha
    sig = p_values < alpha
    classifications = np.full(n, "ns", dtype="<U2")
    classifications[sig & (z > 0) & (Wz > 0)] = "HH"
    classifications[sig & (z < 0) & (Wz < 0)] = "LL"
    classifications[sig & (z > 0) & (Wz < 0)] = "HL"
    classifications[sig & (z < 0) & (Wz > 0)] = "LH"

    # Global I (for reference) = mean(local_I) with row-normalized W
    moran_I_global = float(local_I.mean()) if weights.sum() == n else float(local_I.sum() / weights.sum() * n)

    result = LISAResult(
        local_I=local_I,
        z_attribute=z,
        spatial_lag=Wz,
        p_values=p_values,
        p_values_fdr=p_fdr,
        classifications=classifications,
        alpha=alpha,
        n_permutations=n_permutations,
        bandwidth_km=bandwidth_km,
        attribute_name=attribute_name,
        moran_I_global=moran_I_global,
    )

    duration = time.time() - t0
    if audit:
        run_log.record(
            step="layer2.local_moran",
            duration_s=duration,
            params={
                "attribute_name": attribute_name,
                "bandwidth_km": bandwidth_km,
                "n_permutations": n_permutations,
                "alpha": alpha,
            },
            input_rows=n,
            output_rows=n,
            output_hash=hash_dataframe(result.to_dataframe()),
            notes=result.summary(),
        )

    return result
