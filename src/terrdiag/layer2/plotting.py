"""
terrdiag.layer2.plotting — matplotlib figures for Layer 2 diagnostics.

Stable color palette for LISA classes:
    HH = #d7191c (red)     — high-high (hot spot)
    LL = #2c7bb6 (blue)    — low-low (cold spot)
    HL = #fdae61 (orange)  — high outlier
    LH = #abd9e9 (lightbl) — low outlier
    ns = #cccccc (grey)    — not significant
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from terrdiag.layer2.lisa import LISAResult
    from terrdiag.layer2.moran import MoranResult


LISA_COLORS = {
    "HH": "#d7191c",
    "LL": "#2c7bb6",
    "HL": "#fdae61",
    "LH": "#abd9e9",
    "ns": "#cccccc",
}


def _figure(figsize):
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "terrdiag.layer2.plotting requires matplotlib"
        ) from e
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


# ─── Moran scatter ────────────────────────────────────────────────────────


def plot_moran_scatter(
    result: "MoranResult",
    figsize: tuple[float, float] = (8, 7),
    color_by_classification: "LISAResult | None" = None,
):
    """Moran scatter: standardized attribute × standardized spatial lag.

    Slope of the regression line equals Moran's I.

    Parameters
    ----------
    result : MoranResult
    figsize : tuple
    color_by_classification : LISAResult | None
        If provided, color each point by LISA class.
    """
    fig, ax = _figure(figsize)
    z = result.attribute_values
    Wz = result.spatial_lag

    if color_by_classification is not None:
        cls = color_by_classification.classifications
        for c, color in LISA_COLORS.items():
            mask = cls == c
            if mask.sum() == 0:
                continue
            ax.scatter(z[mask], Wz[mask], c=color, s=35, alpha=0.85,
                        edgecolors="white", linewidth=0.5, label=f"{c} (n={int(mask.sum())})")
    else:
        ax.scatter(z, Wz, s=35, alpha=0.7, c="#444", edgecolors="white", linewidth=0.4)

    # Regression line — slope = I
    xs = np.array([z.min(), z.max()])
    ys = result.I * xs
    ax.plot(xs, ys, color="black", lw=1.5, label=f"slope = I = {result.I:+.3f}")

    ax.axhline(0, color="grey", lw=0.5, alpha=0.5)
    ax.axvline(0, color="grey", lw=0.5, alpha=0.5)
    ax.set_xlabel(f"z({result.attribute_name})")
    ax.set_ylabel(f"spatial lag W·z({result.attribute_name})")
    ax.set_title(
        f"Moran scatterplot — I={result.I:+.4f}, p={result.p_value:.4f} "
        f"(bw={result.bandwidth_km}km, n={result.n})"
    )
    # Quadrant labels
    for q, (sx, sy, txt) in {
        "HH": (0.95, 0.95, "HH"),
        "LH": (0.05, 0.95, "LH"),
        "LL": (0.05, 0.05, "LL"),
        "HL": (0.95, 0.05, "HL"),
    }.items():
        ax.text(sx, sy, txt, transform=ax.transAxes, ha="center", va="center",
                 fontsize=11, weight="bold", color="grey",
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.7))
    ax.legend(loc="upper left", framealpha=0.85, fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


# ─── LISA scatter ─────────────────────────────────────────────────────────


def plot_lisa_scatter(
    result: "LISAResult",
    figsize: tuple[float, float] = (8, 7),
):
    """Moran-like scatter with points colored by LISA classification."""
    fig, ax = _figure(figsize)
    z = result.z_attribute
    Wz = result.spatial_lag

    for cls in ["ns", "LH", "HL", "LL", "HH"]:  # plot ns first so significant points overlay
        mask = result.classifications == cls
        if mask.sum() == 0:
            continue
        ax.scatter(
            z[mask], Wz[mask], c=LISA_COLORS[cls], s=38,
            alpha=0.85, edgecolors="white", linewidth=0.5,
            label=f"{cls} (n={int(mask.sum())})",
        )

    # Slope line: global I (computed as covariance of z and Wz over n)
    xs = np.array([z.min(), z.max()])
    ys = result.moran_I_global * xs
    ax.plot(xs, ys, color="black", lw=1.5, label=f"I = {result.moran_I_global:+.3f}")

    ax.axhline(0, color="grey", lw=0.5, alpha=0.5)
    ax.axvline(0, color="grey", lw=0.5, alpha=0.5)
    ax.set_xlabel(f"z({result.attribute_name})")
    ax.set_ylabel(f"spatial lag W·z({result.attribute_name})")
    ax.set_title(
        f"LISA scatter — α={result.alpha}, bw={result.bandwidth_km}km, n={len(z)}"
    )
    ax.legend(loc="best", framealpha=0.85, fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


# ─── LISA map ──────────────────────────────────────────────────────────────


def plot_lisa_map(
    result: "LISAResult",
    panel: pd.DataFrame,
    x_column: str = "X",
    y_column: str = "Y",
    figsize: tuple[float, float] = (10, 9),
    point_size: int = 50,
):
    """Spatial scatter of LISA classifications."""
    fig, ax = _figure(figsize)
    if x_column not in panel.columns or y_column not in panel.columns:
        raise KeyError(f"panel needs {x_column!r} and {y_column!r}")
    if len(panel) != len(result.local_I):
        raise ValueError("panel/LISA length mismatch")

    # Plot 'ns' first so significant points overlay on top
    for cls in ["ns", "LH", "HL", "LL", "HH"]:
        mask = result.classifications == cls
        if mask.sum() == 0:
            continue
        ax.scatter(
            panel.iloc[mask.nonzero()[0]][x_column],
            panel.iloc[mask.nonzero()[0]][y_column],
            c=LISA_COLORS[cls],
            s=point_size if cls != "ns" else point_size * 0.6,
            alpha=0.85 if cls != "ns" else 0.4,
            edgecolors="black" if cls != "ns" else "white",
            linewidth=0.6 if cls != "ns" else 0.3,
            label=f"{cls} (n={int(mask.sum())})",
        )

    ax.set_xlabel(f"{x_column} (m, EPSG:5880)")
    ax.set_ylabel(f"{y_column} (m, EPSG:5880)")
    ax.set_title(
        f"LISA map — {result.attribute_name}, "
        f"I={result.moran_I_global:+.3f}, "
        f"bw={result.bandwidth_km}km, α={result.alpha}"
    )
    ax.set_aspect("equal")
    ax.legend(loc="best", framealpha=0.85)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


# ─── Classification counts ────────────────────────────────────────────────


def plot_lisa_counts(
    result: "LISAResult",
    figsize: tuple[float, float] = (7, 5),
):
    """Bar chart of LISA classification counts."""
    fig, ax = _figure(figsize)
    classes = ["HH", "LL", "HL", "LH", "ns"]
    counts = {c: int((result.classifications == c).sum()) for c in classes}
    colors = [LISA_COLORS[c] for c in classes]
    ax.bar(classes, [counts[c] for c in classes], color=colors,
            edgecolor="white", linewidth=1.5)
    for c in classes:
        ax.text(c, counts[c], str(counts[c]), ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("count")
    ax.set_title(
        f"LISA classifications — {result.attribute_name} "
        f"(α={result.alpha}, n={len(result.local_I)})"
    )
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


# ─── Bandwidth sensitivity ────────────────────────────────────────────────


def plot_bandwidth_sensitivity(
    sweep_df: pd.DataFrame,
    attribute_name: str = "",
    alpha: float = 0.05,
    figsize: tuple[float, float] = (10, 5),
):
    """Two-panel plot: I and p-value across bandwidths."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    bw = sweep_df["bandwidth_km"].values

    # Panel 1: I
    ax1.plot(bw, sweep_df["I"], marker="o", color="#1b9e77", lw=2)
    ax1.axhline(0, color="grey", lw=0.5, alpha=0.5)
    ax1.set_xlabel("bandwidth (km)")
    ax1.set_ylabel("Moran's I")
    ax1.set_title(f"Moran's I vs bandwidth — {attribute_name}")
    ax1.grid(alpha=0.25)
    for x, y in zip(bw, sweep_df["I"]):
        ax1.text(x, y, f"{y:+.3f}", fontsize=8, ha="center", va="bottom")

    # Panel 2: p-value
    ax2.plot(bw, sweep_df["p_value"], marker="o", color="#d95f02", lw=2)
    ax2.axhline(alpha, color="red", lw=0.8, linestyle="--",
                 label=f"α = {alpha}")
    ax2.set_xlabel("bandwidth (km)")
    ax2.set_ylabel("p-value (permutation)")
    ax2.set_title(f"p-value vs bandwidth — {attribute_name}")
    ax2.set_yscale("log")
    ax2.grid(alpha=0.25, which="both")
    ax2.legend()

    fig.tight_layout()
    return fig
