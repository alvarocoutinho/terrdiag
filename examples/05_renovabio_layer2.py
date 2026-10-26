"""
05_renovabio_layer2.py — Layer 2 LISA on RenovaBio NEEA dynamics.

Operates on the mill-level canonical panel (n=230) with X, Y projected coords.
Tests global spatial autocorrelation of ΔNEEA and identifies local hot/cold
spots, plus a bandwidth sensitivity sweep.

Usage:
    python examples/05_renovabio_layer2.py --panel /path/to/painel.csv \\
        --output-dir ./figs_layer2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from terrdiag import audit, io, layer2
from terrdiag.case_studies.renovabio.make_panel import RENOVABIO_SCHEMA
from terrdiag.case_studies.renovabio.presets import RENOVABIO_OUTCOMES


def main(panel_path: str, output_dir: str, bandwidth_km: float = 100.0) -> int:
    audit.run_log.reset()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    figs_saved: list[str] = []

    def _save(name, fig):
        path = out / f"{name}.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        figs_saved.append(str(path))

    print("═" * 70)
    print(f"Layer 2 LISA pipeline (bw={bandwidth_km} km)")
    print("═" * 70)

    # 1. Load panel + filter to mills with geo + ΔNEEA
    panel = io.load_panel(panel_path, schema=RENOVABIO_SCHEMA, outcomes=RENOVABIO_OUTCOMES)
    mill = panel[
        panel["X"].notna() & panel["Y"].notna() & panel["delta_neea"].notna()
    ].reset_index(drop=True)
    print(f"\n[1] {len(mill)} mills with geo + ΔNEEA "
          f"(of {len(panel)} in panel)")

    # 2. Weights (Gaussian kernel)
    X, Y = mill["X"].to_numpy(), mill["Y"].to_numpy()
    W = layer2.gaussian_kernel_weights(X, Y, bandwidth_km=bandwidth_km)
    n_nb = layer2.n_neighbors_per_row(W, threshold=1e-6)
    print(f"\n[2] Weights matrix: {W.shape}")
    print(f"    Effective neighbors per mill: min={n_nb.min()}, "
          f"mean={n_nb.mean():.1f}, max={n_nb.max()}")

    # 3. Global Moran on ΔNEEA, NEEA t0, NEEA t2
    print("\n[3] Global Moran's I")
    print("-" * 70)
    morans = {}
    for col, name in [
        ("delta_neea", "ΔNEEA"),
        ("neea_t0_2022", "NEEA t0_2022"),
        ("neea_t2_2026", "NEEA t2_2026"),
    ]:
        s = mill[col]
        mask = s.notna()
        Wm = layer2.gaussian_kernel_weights(
            mill.loc[mask, "X"].to_numpy(),
            mill.loc[mask, "Y"].to_numpy(),
            bandwidth_km=bandwidth_km,
        )
        m = layer2.moran_i(
            s[mask].to_numpy(), Wm,
            n_permutations=999, attribute_name=name, bandwidth_km=bandwidth_km,
        )
        morans[col] = m
        print(f"  {m.summary()}")
        _save(f"moran_{col}", m.plot())

    # 4. LISA on ΔNEEA
    print("\n[4] LISA — ΔNEEA")
    print("-" * 70)
    lisa = layer2.local_moran(
        mill["delta_neea"].to_numpy(), W,
        n_permutations=999, attribute_name="ΔNEEA", bandwidth_km=bandwidth_km,
    )
    print(f"  {lisa.summary()}")
    print()
    print(layer2.lisa_classification_counts(lisa).round(3).to_string())

    _save("lisa_scatter", lisa.plot(kind="scatter"))
    _save("lisa_map", lisa.plot(kind="map", panel=mill))
    _save("lisa_counts", lisa.plot(kind="counts"))

    # 5. LISA × UF
    print("\n[5] LISA × UF")
    print("-" * 70)
    print(layer2.lisa_by_group(lisa, mill, "uf").to_string())

    # 6. Top contributors
    print("\n[6] TOP 5 HH (hot spots — ΔNEEA accelerated)")
    print("-" * 70)
    top_hh = layer2.lisa_top_contributors(lisa, mill, n=5, classification="HH")
    if len(top_hh):
        print(top_hh[["cnpj_clean", "emissor", "uf", "delta_neea", "local_I", "p_value"]]
                .to_string(index=False))

    print("\n[6] TOP 5 LL (cold spots — ΔNEEA decreased)")
    print("-" * 70)
    top_ll = layer2.lisa_top_contributors(lisa, mill, n=5, classification="LL")
    if len(top_ll):
        print(top_ll[["cnpj_clean", "emissor", "uf", "delta_neea", "local_I", "p_value"]]
                .to_string(index=False))

    # 7. Bandwidth sensitivity
    print("\n[7] Bandwidth sensitivity sweep")
    print("-" * 70)
    sweep = layer2.bandwidth_sweep(
        mill["delta_neea"].to_numpy(), X, Y,
        bandwidths_km=[50, 75, 100, 125, 150, 200],
        n_permutations=499, attribute_name="ΔNEEA",
    )
    print(sweep.round(4).to_string(index=False))
    _save("sensitivity", layer2.plot_sensitivity(sweep, attribute_name="ΔNEEA"))

    # 8. Significance diagnostics
    print("\n[8] Significance diagnostics")
    print("-" * 70)
    print(layer2.lisa_significance_diagnostics(lisa).round(4).to_string(index=False))

    # 9. Audit
    print("\n[9] Audit log")
    print("-" * 70)
    for e in audit.run_log.entries:
        print(f"  [{e.duration_s:.2f}s] {e.step}")

    print(f"\nFigures saved ({len(figs_saved)}):")
    for f in figs_saved:
        print(f"  {f}")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel", required=True)
    p.add_argument("--output-dir", default="figs_layer2")
    p.add_argument("--bandwidth-km", type=float, default=100.0)
    args = p.parse_args()
    sys.exit(main(args.panel, args.output_dir, args.bandwidth_km))
