"""
terrdiag.overlay — Cross-tabulation of Layer 1 (municipal cluster) ×
Layer 2 (mill LISA class).

The two-layer framework's payoff. Each mill is jointly classified by:
  - territorial typology (Layer 1)
  - local spatial regime (Layer 2)

producing diagnostic cells that connect macro-structural with micro-spatial
information. Cells like C2_frontier × HH or C0_intensive × LL are
interpretable units for policy narration.

Public API:

  diagnostic_cells(layer1_result, layer2_result, layer1_panel, layer2_panel,
                    join_column='cd_ibge', cluster_names=None, ...)
    → OverlayResult

  OverlayResult:
    .summary() / .counts(normalize=...) / .cell(cluster, lisa_class) /
    .plot(kind='heatmap'|'sankey'|'map') / .to_dict()

  Inspection:
    cell_summary, cell_representatives, cluster_lisa_marginals,
    association_test (chi-squared, Cramér's V)

  Plotting:
    plot_overlay_heatmap, plot_overlay_sankey, plot_overlay_map

Typical pipeline:

    from terrdiag import overlay

    result_overlay = overlay.diagnostic_cells(
        layer1_result, layer2_result,
        layer1_panel=muni_panel, layer2_panel=mill_panel,
        cluster_names={
            0: 'C0_agroindustrial_intensive',
            1: 'C1_developed_diversified',
            2: 'C2_extensive_vulnerable_frontier',
        },
        bandwidth_metadata={'method': 'knn', 'k': 6},
    )

    print(result_overlay.summary())
    print(result_overlay.counts())                # 3×5 raw counts
    print(result_overlay.counts(normalize='cluster'))  # row shares
    print(overlay.cell_summary(result_overlay))        # long table
    print(overlay.association_test(result_overlay))    # χ² independence

    result_overlay.plot(kind='heatmap')
    result_overlay.plot(kind='sankey')
    result_overlay.plot(kind='map')

    # exemplary mills in C2_frontier × HH cell:
    print(overlay.cell_representatives(
        result_overlay, 'C2_extensive_vulnerable_frontier', 'HH', n=5,
    ))
"""

from terrdiag.overlay.diagnostic import OverlayResult, diagnostic_cells
from terrdiag.overlay.inspect import (
    association_test,
    cell_representatives,
    cell_summary,
    cluster_lisa_marginals,
)
from terrdiag.overlay.plotting import (
    plot_overlay_heatmap,
    plot_overlay_map,
    plot_overlay_map_geo,
    plot_overlay_sankey,
)
from terrdiag.overlay import inspect, plotting

__all__ = [
    "OverlayResult",
    "diagnostic_cells",
    "association_test",
    "cell_representatives",
    "cell_summary",
    "cluster_lisa_marginals",
    "plot_overlay_heatmap",
    "plot_overlay_map",
    "plot_overlay_map_geo",
    "plot_overlay_sankey",
    "inspect",
    "plotting",
]
