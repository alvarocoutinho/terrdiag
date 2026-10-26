"""
terrdiag.layer1.cluster — K-means territorial typology with bootstrap ARI.

Pipeline (frozen for reproducibility):
    StandardScaler → PCA(n=pca_components) → KMeans(k=k_clusters)

Bootstrap stability:
    For each of n_bootstrap iterations, sample rows with replacement, fit
    the full pipeline, compute Adjusted Rand Index against the reference
    fit. Returns the ARI distribution.

═══════════════════════════════════════════════════════════════════════════
NEEA GUARD-RAIL — methodological invariant
═══════════════════════════════════════════════════════════════════════════
Layer 1 typology must be **strictly independent** of the outcome variable
(NEEA, vol_pct). If the outcome leaked into the feature set, the post-hoc
Kruskal-Wallis test of independence (terrdiag.layer1.independence) would
be circular.

`cluster()` therefore inspects feature names BEFORE fitting and raises
ValueError if any name matches `r'neea|vol_pct|vol_eleg|cbio'` (case
insensitive). Override via `_allow_outcome_features=True` only for testing.
═══════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from terrdiag.audit.hashing import hash_dataframe
from terrdiag.audit.logging import run_log
from terrdiag.specs import FeatureRecipe, FeatureSpec


# ─── Forbidden patterns — outcome variables that must NOT be in features ──
OUTCOME_FORBIDDEN_PATTERNS = re.compile(r"(neea|vol_pct|vol_eleg|cbio)", re.IGNORECASE)


def _validate_no_outcome_in_features(features: Iterable[FeatureSpec]) -> None:
    """Raise ValueError if any feature name/source_column contains an outcome substring."""
    violations: list[str] = []
    for f in features:
        for attr_val in (f.name, f.source_column):
            if attr_val and OUTCOME_FORBIDDEN_PATTERNS.search(str(attr_val)):
                violations.append(f"{f.name} (source_column={f.source_column})")
                break
    if violations:
        raise ValueError(
            "Layer 1 GUARD-RAIL violation: outcome variables (NEEA, vol_pct, vol_eleg, "
            "cbio) MUST NOT appear in the feature set used for territorial typology.\n"
            f"Offending features:\n  - " + "\n  - ".join(violations)
            + "\n\nIf this is intentional (e.g. testing), pass _allow_outcome_features=True."
        )


# ─── Result container ──────────────────────────────────────────────────────


@dataclass(slots=True)
class ClusterResult:
    """Output of `cluster()`.

    Attributes
    ----------
    labels : np.ndarray
        Cluster label for each row in the input df, integer in [0, k).
    pca_components : np.ndarray
        PCA-projected coordinates (n_rows × pca_components).
    pca_loadings : np.ndarray
        PCA loadings (n_features × pca_components). Rows aligned with
        `feature_names`. Used for feature-importance plots.
    centroids : np.ndarray
        Cluster centroids in PCA space (k × pca_components).
    inertia : float
        K-means within-cluster sum of squares.
    silhouette : float
        Silhouette score on the PCA-projected data.
    ari_bootstrap : np.ndarray
        ARI values across n_bootstrap iterations.
    feature_names : tuple[str, ...]
        Features used (after dropping high-missingness columns).
    feature_source_columns : tuple[str, ...]
        Aligned source-column names used in the original panel.
    n_dropped_features : int
        How many features were dropped due to missing_threshold.
    explained_variance_ratio : np.ndarray
        PCA explained variance per component.
    random_state : int

    Methods
    -------
    summary()       — One-line summary string.
    profile(panel)  — Mean/median by cluster per feature (DataFrame).
    loadings()      — Per-feature loadings (DataFrame).
    plot(kind=...)  — Render figures: 'pca', 'centroids', 'loadings',
                      'profile', 'ari', 'silhouette', 'by_uf', 'map'.
    report(panel)   — Dict containing all numerical artifacts.
    """

    labels: np.ndarray
    pca_components: np.ndarray
    pca_loadings: np.ndarray
    centroids: np.ndarray
    inertia: float
    silhouette: float
    ari_bootstrap: np.ndarray
    feature_names: tuple[str, ...]
    feature_source_columns: tuple[str, ...]
    n_dropped_features: int
    explained_variance_ratio: np.ndarray
    random_state: int

    def summary(self) -> str:
        ari = self.ari_bootstrap
        return (
            f"ClusterResult: k={len(self.centroids)}, "
            f"n_features={len(self.feature_names)} (dropped {self.n_dropped_features}), "
            f"silhouette={self.silhouette:.3f}, "
            f"ARI bootstrap mean={ari.mean():.3f} (CI95=[{np.percentile(ari, 2.5):.3f}, "
            f"{np.percentile(ari, 97.5):.3f}]), "
            f"PCA explained={self.explained_variance_ratio.sum():.1%}"
        )

    def profile(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Mean and median of each feature by cluster. Wide DataFrame."""
        from terrdiag.layer1.inspect import cluster_profile
        return cluster_profile(self, panel)

    def loadings(self) -> pd.DataFrame:
        """PCA loadings as a DataFrame (n_features × n_components)."""
        from terrdiag.layer1.inspect import feature_loadings
        return feature_loadings(self)

    def plot(self, kind: str = "pca", **kwargs):
        """Render a diagnostic figure.

        Available kinds: 'pca', 'centroids', 'loadings', 'profile',
        'ari', 'silhouette', 'by_uf', 'map'.
        """
        from terrdiag.layer1 import plotting
        dispatch = {
            "pca": plotting.plot_pca_scatter,
            "centroids": plotting.plot_centroids,
            "loadings": plotting.plot_feature_loadings,
            "profile": plotting.plot_cluster_profile,
            "ari": plotting.plot_ari_bootstrap,
            "silhouette": plotting.plot_silhouette_profile,
            "by_uf": plotting.plot_cluster_by_group,
            "map": plotting.plot_xy_map,
        }
        if kind not in dispatch:
            raise ValueError(
                f"Unknown plot kind: {kind!r}. Available: {sorted(dispatch.keys())}"
            )
        return dispatch[kind](self, **kwargs)

    def report(self, panel: pd.DataFrame | None = None) -> dict:
        """Comprehensive diagnostic dict suitable for JSON serialization or markdown."""
        rep = {
            "summary": self.summary(),
            "n_clusters": int(len(self.centroids)),
            "n_entities": int(len(self.labels)),
            "n_features": int(len(self.feature_names)),
            "n_dropped_features": int(self.n_dropped_features),
            "silhouette": float(self.silhouette),
            "inertia": float(self.inertia),
            "ari_bootstrap_mean": float(self.ari_bootstrap.mean()),
            "ari_bootstrap_ci95": [
                float(np.percentile(self.ari_bootstrap, 2.5)),
                float(np.percentile(self.ari_bootstrap, 97.5)),
            ],
            "pca_explained_variance_ratio": [float(v) for v in self.explained_variance_ratio],
            "feature_names": list(self.feature_names),
            "cluster_sizes": {int(k): int(v) for k, v in
                              zip(*np.unique(self.labels, return_counts=True))},
        }
        if panel is not None:
            rep["cluster_profile"] = self.profile(panel).to_dict()
        return rep


# ─── Main entrypoint ───────────────────────────────────────────────────────


def cluster(
    panel: pd.DataFrame,
    recipe: FeatureRecipe,
    audit: bool = True,
    _allow_outcome_features: bool = False,
) -> ClusterResult:
    """Run the canonical Layer 1 K-means pipeline.

    Pre-conditions:
        - All features in `recipe.features` are columns in `panel`.
        - `panel` has been preprocessed (transforms, clips, imputation already applied).
          Call `terrdiag.layer1.preprocess` and `terrdiag.layer1.impute` first.

    Parameters
    ----------
    panel : pd.DataFrame
        Preprocessed entity-level panel (one row per entity).
    recipe : FeatureRecipe
        Frozen recipe with features + clustering parameters.
    audit : bool
        Emit a run-log entry.
    _allow_outcome_features : bool
        Disable the NEEA guard-rail. Use ONLY in tests of independence
        function. Production code should never set this.

    Returns
    -------
    ClusterResult

    Raises
    ------
    ValueError
        If guard-rail detects an outcome variable in features (unless
        _allow_outcome_features=True).
    """
    try:
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        from sklearn.metrics import adjusted_rand_score, silhouette_score
        from sklearn.preprocessing import RobustScaler, StandardScaler
    except ImportError as e:
        raise ImportError(
            "terrdiag.layer1.cluster requires scikit-learn. "
            "Install with: pip install 'terrdiag[layer1]'"
        ) from e

    t0 = time.time()

    # ═══ GUARD-RAIL ═══
    if not _allow_outcome_features:
        _validate_no_outcome_in_features(recipe.features)

    # Build feature matrix
    feature_source_cols = [s.source_column for s in recipe.features]
    missing_cols = [c for c in feature_source_cols if c not in panel.columns]
    if missing_cols:
        raise KeyError(
            f"Features missing from panel: {missing_cols}. "
            f"Did you call apply_recipe to materialize the recipe?"
        )

    # Drop features exceeding missingness threshold
    kept_specs: list[FeatureSpec] = []
    dropped: list[str] = []
    for spec in recipe.features:
        miss_frac = panel[spec.source_column].isna().mean()
        if miss_frac > spec.missing_threshold:
            dropped.append(f"{spec.name} ({miss_frac:.1%} missing)")
        else:
            kept_specs.append(spec)

    if len(kept_specs) < recipe.pca_components:
        raise ValueError(
            f"After dropping high-missingness features, only {len(kept_specs)} remain "
            f"but pca_components={recipe.pca_components}. Lower the threshold "
            f"or PCA components."
        )

    kept_cols = [s.source_column for s in kept_specs]
    X = panel[kept_cols].to_numpy(dtype=float)

    if np.isnan(X).any():
        raise ValueError(
            f"Feature matrix still contains NaN after dropping high-missingness columns. "
            f"Call terrdiag.layer1.impute first (or set missing_threshold lower)."
        )

    # Pipeline
    scaler_cls = {"standard": StandardScaler, "robust": RobustScaler}.get(recipe.scaler)
    if scaler_cls is None and recipe.scaler != "none":
        raise ValueError(f"Unknown scaler: {recipe.scaler}")
    Xs = scaler_cls().fit_transform(X) if scaler_cls is not None else X.copy()

    pca = PCA(n_components=recipe.pca_components, random_state=recipe.random_state)
    Xp = pca.fit_transform(Xs)

    km = KMeans(
        n_clusters=recipe.k_clusters,
        random_state=recipe.random_state,
        n_init=10,
    )
    labels = km.fit_predict(Xp)

    sil = float(silhouette_score(Xp, labels))

    # Bootstrap ARI
    rng = np.random.default_rng(recipe.random_state)
    n = len(Xp)
    ari_values = np.empty(recipe.n_bootstrap, dtype=float)
    for i in range(recipe.n_bootstrap):
        idx = rng.integers(0, n, size=n)
        Xb = Xs[idx]
        # Apply same PCA fit on the bootstrap sample (fit fresh for stability test)
        pca_b = PCA(n_components=recipe.pca_components, random_state=recipe.random_state)
        Xpb = pca_b.fit_transform(Xb)
        km_b = KMeans(
            n_clusters=recipe.k_clusters,
            random_state=recipe.random_state + i,
            n_init=10,
        )
        labels_b = km_b.fit_predict(Xpb)
        # Compare bootstrap labels on resampled idx vs reference labels at those positions
        ari_values[i] = adjusted_rand_score(labels[idx], labels_b)

    result = ClusterResult(
        labels=labels,
        pca_components=Xp,
        pca_loadings=pca.components_.T,  # n_features × n_components
        centroids=km.cluster_centers_,
        inertia=float(km.inertia_),
        silhouette=sil,
        ari_bootstrap=ari_values,
        feature_names=tuple(s.name for s in kept_specs),
        feature_source_columns=tuple(s.source_column for s in kept_specs),
        n_dropped_features=len(dropped),
        explained_variance_ratio=pca.explained_variance_ratio_,
        random_state=recipe.random_state,
    )

    duration = time.time() - t0
    if audit:
        run_log.record(
            step="layer1.cluster",
            duration_s=duration,
            params={
                "recipe_name": recipe.name,
                "k": recipe.k_clusters,
                "pca_components": recipe.pca_components,
                "n_features_input": len(recipe.features),
                "n_features_used": len(kept_specs),
                "n_features_dropped": len(dropped),
                "scaler": recipe.scaler,
                "random_state": recipe.random_state,
                "n_bootstrap": recipe.n_bootstrap,
            },
            input_rows=len(panel),
            output_rows=len(labels),
            output_hash=hash_dataframe(pd.DataFrame({"label": labels})),
            notes=result.summary() + (f" | dropped: {dropped}" if dropped else ""),
        )

    return result
