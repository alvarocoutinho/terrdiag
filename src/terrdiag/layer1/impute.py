"""
terrdiag.layer1.impute — Missing-value imputation for Layer 1 features.

The default strategy is `median_by_group` with fallback to global median.
For RenovaBio, the group column is 'uf' (state). Imputation is applied
column-by-column on the feature subset specified.

Why median: robust to outliers, preserves bounded ranges (e.g. shares in
[0,1]), and matches the v0.4 alpha behavior.
"""

from __future__ import annotations

import time
from typing import Iterable, Literal

import pandas as pd

from terrdiag.audit.hashing import hash_dataframe
from terrdiag.audit.logging import run_log


ImputationStrategy = Literal[
    "median_by_group", "median_global", "mean_global", "drop"
]


def impute(
    panel: pd.DataFrame,
    columns: Iterable[str],
    strategy: ImputationStrategy = "median_by_group",
    group_column: str | None = "uf",
    audit: bool = True,
) -> pd.DataFrame:
    """Fill missing values in selected columns.

    Parameters
    ----------
    panel : pd.DataFrame
    columns : iterable of str
        Columns to impute (typically the feature subset for clustering).
    strategy : {'median_by_group', 'median_global', 'mean_global', 'drop'}
        How to fill missing values.
        - 'median_by_group': median within `group_column`, fallback global median
        - 'median_global':   global median across all rows
        - 'mean_global':     global mean
        - 'drop':            drop rows with any missing in the selected columns
    group_column : str | None
        Column for grouping (only used by 'median_by_group').
    audit : bool

    Returns
    -------
    pd.DataFrame
        Copy with missing values filled (or rows dropped).
    """
    t0 = time.time()
    out = panel.copy()
    columns = list(columns)

    n_missing_before = int(out[columns].isna().sum().sum())

    if strategy == "drop":
        out = out.dropna(subset=columns).reset_index(drop=True)
    elif strategy == "median_global":
        for c in columns:
            out[c] = out[c].fillna(out[c].median())
    elif strategy == "mean_global":
        for c in columns:
            out[c] = out[c].fillna(out[c].mean())
    elif strategy == "median_by_group":
        if group_column is None or group_column not in out.columns:
            raise ValueError(
                f"strategy='median_by_group' requires group_column in panel; "
                f"got {group_column!r}, available: {sorted(out.columns)[:10]}..."
            )
        for c in columns:
            global_med = out[c].median()
            # Group-level medians
            group_med = out.groupby(group_column)[c].transform("median")
            out[c] = out[c].fillna(group_med).fillna(global_med)
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")

    n_missing_after = int(out[columns].isna().sum().sum())

    duration = time.time() - t0
    if audit:
        run_log.record(
            step="layer1.impute",
            duration_s=duration,
            params={
                "strategy": strategy,
                "group_column": group_column,
                "n_columns": len(columns),
            },
            input_rows=len(panel),
            output_rows=len(out),
            output_hash=hash_dataframe(out),
            notes=(
                f"missing before: {n_missing_before}, after: {n_missing_after}; "
                f"rows: {len(panel)} → {len(out)}"
            ),
        )

    return out
