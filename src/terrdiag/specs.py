"""
terrdiag.specs — Configuration dataclasses (immutable).

Each spec is a frozen dataclass that travels with the data through the pipeline.
Specs are the contract between recipes and pipeline functions.

Mutating a spec mid-pipeline raises FrozenInstanceError — by design.

Convention: spec instances passed to pipeline functions become part of the
run metadata emitted by terrdiag.audit. Two runs with identical specs and
inputs must produce identical outputs.

Only generic specs live here. Case-specific specs (e.g. snapshot file
descriptors for ANP) belong to the relevant case_studies/ subfolder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# ─────────────────────────────────────────────────────────────────────────────
# Layer 1 — feature specs (entity-level features for K-means)
# ─────────────────────────────────────────────────────────────────────────────

TransformKind = Literal["identity", "log1p", "zscore", "percentile_rank"]
ImputationKind = Literal["median_by_group", "median_global", "mean_global", "drop"]
ScalerKind = Literal["standard", "robust", "none"]


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """One feature in the Layer 1 (entity-level) clustering recipe.

    Attributes
    ----------
    name : str
        Canonical short name (used in output columns and figures).
    source_column : str
        Column name in the panel DataFrame.
    transform : str
        Pre-scaling transform. 'identity' applies nothing.
    clip : tuple[float, float] | None
        Optional (min, max) clipping after compute, before scaling.
        Useful for share/proportion features that must be in [0, 1].
    missing_threshold : float
        Drop the feature if missingness exceeds this fraction (0-1).
    description : str
        Human-readable description, for docs and audit logs.
    dimension : str
        Optional thematic dimension (e.g. 'Economy', 'Land') used by
        downstream plotting and radar diagrams.
    """

    name: str
    source_column: str
    transform: TransformKind = "identity"
    clip: tuple[float, float] | None = None
    missing_threshold: float = 0.20
    description: str = ""
    dimension: str = ""


@dataclass(frozen=True, slots=True)
class FeatureRecipe:
    """A complete Layer 1 recipe: features + pipeline parameters."""

    name: str
    features: tuple[FeatureSpec, ...]
    imputation: ImputationKind = "median_by_group"
    impute_group_column: str | None = "uf"
    scaler: ScalerKind = "standard"
    pca_components: int = 2
    k_clusters: int = 3
    random_state: int = 42
    n_bootstrap: int = 200
    description: str = ""

    def __post_init__(self) -> None:
        if not self.features:
            raise ValueError("FeatureRecipe must have at least one feature")
        if self.pca_components < 1:
            raise ValueError(f"pca_components must be ≥1, got {self.pca_components}")
        if self.k_clusters < 2:
            raise ValueError(f"k_clusters must be ≥2, got {self.k_clusters}")

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.features)


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2 — outcome specs (longitudinal metrics measured at event points)
# ─────────────────────────────────────────────────────────────────────────────

OutcomeDirection = Literal["higher_better", "lower_better", "neutral"]


@dataclass(frozen=True, slots=True)
class OutcomeSpec:
    """One longitudinal outcome metric for Layer 2 (LISA).

    Attributes
    ----------
    name : str
        Canonical name used in figures and result columns.
    snapshots : dict[str, str]
        Mapping snapshot_label → column name in panel. Convention keys
        are 't0', 't1', 't2' (snapshot order); the actual column names
        are user-defined.
    delta_column : str
        Column name with the cohort-aware delta in the panel.
    first_column : str
        Column name with the cohort-aware initial value.
    first_origin_column : str
        Column name with the label of the snapshot used as initial reference.
    units : str
        Display units (e.g. 'gCO2eq/MJ', '%', 'score').
    direction : str
        Policy interpretation: 'higher_better', 'lower_better', or 'neutral'.
    description : str
        Human-readable description.

    Notes
    -----
    The cohort-aware delta is expected to be computed at panel-construction
    time (outside terrdiag, typically in case_studies/<case>/make_panel.py).
    OutcomeSpec just points to the columns.
    """

    name: str
    snapshots: dict[str, str] = field(default_factory=dict)
    delta_column: str = ""
    first_column: str = ""
    first_origin_column: str = ""
    units: str = ""
    direction: OutcomeDirection = "neutral"
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("OutcomeSpec.name is required")
        if not self.delta_column:
            raise ValueError(f"OutcomeSpec({self.name}).delta_column is required")

    @property
    def required_columns(self) -> tuple[str, ...]:
        """All columns this outcome expects to find in the panel."""
        cols = list(self.snapshots.values())
        if self.delta_column:
            cols.append(self.delta_column)
        if self.first_column:
            cols.append(self.first_column)
        if self.first_origin_column:
            cols.append(self.first_origin_column)
        return tuple(cols)


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2 — LISA recipe (kernel, bandwidth, permutations)
# ─────────────────────────────────────────────────────────────────────────────

KernelKind = Literal["gaussian", "uniform", "triangular", "epanechnikov"]


@dataclass(frozen=True, slots=True)
class LISARecipe:
    """Configuration for LISA computation (kernel weights + Moran + local I)."""

    kernel: KernelKind = "gaussian"
    bandwidth_km: float = 100.0
    sensitivity_bandwidths_km: tuple[float, ...] = (50.0, 100.0, 150.0)
    permutations: int = 999
    random_state: int = 42
    significance_level: float = 0.05

    def __post_init__(self) -> None:
        if self.bandwidth_km <= 0:
            raise ValueError(f"bandwidth_km must be > 0, got {self.bandwidth_km}")
        if self.permutations < 99:
            raise ValueError(
                f"permutations must be ≥99 for valid inference, got {self.permutations}"
            )
        if not 0 < self.significance_level < 1:
            raise ValueError(
                f"significance_level must be in (0, 1), got {self.significance_level}"
            )
