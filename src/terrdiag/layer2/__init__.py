"""
terrdiag.layer2 — Spatial autocorrelation of mill-level outcomes via LISA.

Operates on event points (mills with X, Y projected coordinates) rather than
on entity-level aggregates. Outcome variables (NEEA, vol_pct, delta_neea, ...)
are passed explicitly — there is no risk of feature/outcome leakage because
Layer 2 is, by design, outcome-focused.

Public API:

  Weights:
    - gaussian_kernel_weights(X, Y, bandwidth_km) → (n, n) ndarray
    - knn_weights(X, Y, k)                        → (n, n) ndarray

  Global statistic:
    - moran_i(values, weights, ...) → MoranResult
      MoranResult: .summary() / .plot() / .to_dict()

  Local statistic:
    - local_moran(values, weights, ...) → LISAResult
      LISAResult: .summary() / .classify() / .to_dataframe(panel) /
                  .plot(kind='scatter'|'map'|'counts') / .to_dict()

  Sensitivity:
    - bandwidth_sweep(values, X, Y, bandwidths_km) → DataFrame
    - plot_sensitivity(sweep_df)                     → Figure

  Inspection (pure):
    - lisa_classification_counts, lisa_top_contributors, lisa_by_group,
    - lisa_significance_diagnostics, moran_permutation_diagnostics

  Plotting (matplotlib):
    - plot_moran_scatter, plot_lisa_scatter, plot_lisa_map,
    - plot_lisa_counts, plot_bandwidth_sensitivity

Typical pipeline:

    from terrdiag.layer2 import (
        gaussian_kernel_weights, moran_i, local_moran, bandwidth_sweep
    )

    W = gaussian_kernel_weights(panel['X'], panel['Y'], bandwidth_km=100)
    mi = moran_i(panel['delta_neea'], W, attribute_name='ΔNEEA', bandwidth_km=100)
    lisa = local_moran(panel['delta_neea'], W, attribute_name='ΔNEEA', bandwidth_km=100)

    mi.plot()                       # Moran scatter
    lisa.plot(kind='map', panel=panel)
    lisa.plot(kind='scatter')
    lisa.plot(kind='counts')

    sweep = bandwidth_sweep(panel['delta_neea'], panel['X'], panel['Y'],
                              bandwidths_km=[50, 75, 100, 125, 150])
"""

from terrdiag.layer2.lisa import LISAResult, local_moran
from terrdiag.layer2.moran import MoranResult, moran_i
from terrdiag.layer2.sensitivity import bandwidth_sweep, plot_sensitivity
from terrdiag.layer2.weights import (
    gaussian_kernel_weights,
    inverse_distance_weights,
    knn_weights,
    n_neighbors_per_row,
)
from terrdiag.layer2 import inspect, plotting
from terrdiag.layer2.inspect import (
    lisa_by_group,
    lisa_classification_counts,
    lisa_significance_diagnostics,
    lisa_top_contributors,
    moran_permutation_diagnostics,
)
from terrdiag.layer2.plotting import (
    plot_bandwidth_sensitivity,
    plot_lisa_counts,
    plot_lisa_map,
    plot_lisa_scatter,
    plot_moran_scatter,
)

__all__ = [
    # core
    "MoranResult",
    "LISAResult",
    "moran_i",
    "local_moran",
    "gaussian_kernel_weights",
    "inverse_distance_weights",
    "knn_weights",
    "n_neighbors_per_row",
    "bandwidth_sweep",
    "plot_sensitivity",
    # submodules
    "inspect",
    "plotting",
    # inspect
    "lisa_by_group",
    "lisa_classification_counts",
    "lisa_significance_diagnostics",
    "lisa_top_contributors",
    "moran_permutation_diagnostics",
    # plotting
    "plot_bandwidth_sensitivity",
    "plot_lisa_counts",
    "plot_lisa_map",
    "plot_lisa_scatter",
    "plot_moran_scatter",
]
