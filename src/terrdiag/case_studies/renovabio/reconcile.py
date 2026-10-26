"""
terrdiag.case_studies.renovabio.reconcile — Opção 2 reconciliation + wide pivot.

Two functions:
    - reconcile_and_filter: apply geographic / route / biofuel filters and
      pick canonical certificates per (cnpj, snapshot) using the Opção 2
      logic (valido OR cancelado-por-renovacao).
    - to_wide_with_deltas: pivot the long panel to wide, compute cohort-aware
      first values and deltas for the declared outcomes.

The output of `to_wide_with_deltas` is a panel that meets terrdiag's
canonical wide schema.
"""

from __future__ import annotations

import time
from typing import Iterable, Literal

import numpy as np
import pandas as pd

from terrdiag.audit.hashing import hash_dataframe
from terrdiag.audit.logging import run_log
from terrdiag.specs import OutcomeSpec


LONG_COLUMNS: tuple[str, ...] = (
    "cnpj_clean",
    "snapshot_date",
    "snapshot_label",
    "cohort",
    "emissor",
    "cidade",
    "uf",
    "rota",
    "rota_tag",
    "processo",
    "status",
    "cat_motivo",
    "priority",
    "observable",
    "data_aprov",
    "neea",
    "vol_pct",
    "in_panel",
)


def _rota_tag(r: object) -> str | None:
    if pd.isna(r):
        return None
    return "cana_milho_integrada" if "milho em usina integrada" in str(r) else "cana_pura"


def reconcile_and_filter(
    raw: pd.DataFrame,
    geography: Iterable[str],
    routes: Iterable[str],
    biofuel: str = "etanol hidratado",
    option: Literal["strict", "reconciled_v2"] = "reconciled_v2",
    audit: bool = True,
) -> pd.DataFrame:
    """Apply RenovaBio-specific filters and reconciliation.

    Parameters
    ----------
    raw : pd.DataFrame
        Output of `load_anp_raw`.
    geography : iterable of str
        Allowed UF codes (e.g. Centro-Sul: SP, GO, MG, PR, MS, MT).
    routes : iterable of str
        Allowed Rota strings (e.g. E1GC cana 1G variants).
    biofuel : str
        Biocombustível filter (case-insensitive).
    option : {'strict', 'reconciled_v2'}
        'strict' = only status == 'valido'.
        'reconciled_v2' = valid OR cancelled-by-renewal (Opção 2 spec).
    audit : bool
        Emit a run-log entry.

    Returns
    -------
    pd.DataFrame
        Long-format reconciled panel: one row per (cnpj_clean, snapshot_label)
        for mills meeting the cohort-aware criteria.

    Notes
    -----
    Cohort-aware criterion: mill is in panel ⇔ observable at the latest
    snapshot AND observable at any earlier snapshot.
    "Observable" = priority ≥ 1 (valid for 'strict'; valid OR canc-renov
    for 'reconciled_v2').
    """
    t_start = time.time()
    geography = list(geography)
    routes = list(routes)

    universe = raw[
        raw["uf"].isin(geography)
        & raw["rota"].isin(routes)
        & (raw["biocombustivel"].str.lower().str.strip() == biofuel)
    ].copy()

    if option == "strict":
        universe["priority"] = (universe["status"] == "valido").astype(int) * 2
    elif option == "reconciled_v2":
        universe["priority"] = (
            (universe["status"] == "valido").astype(int) * 2
            + (
                (universe["status"] == "cancelado")
                & (universe["cat_motivo"] == "renovacao")
            ).astype(int)
        )
    else:
        raise ValueError(f"option must be 'strict' or 'reconciled_v2', got {option!r}")

    universe["observable"] = universe["priority"] >= 1

    universe_sorted = universe.sort_values(
        ["cnpj_clean", "snapshot_label", "priority", "data_aprov"],
        ascending=[True, True, False, False],
    )
    canon = universe_sorted.drop_duplicates(
        subset=["cnpj_clean", "snapshot_label"], keep="first"
    ).copy()

    pivot = canon.pivot_table(
        index="cnpj_clean",
        columns="snapshot_label",
        values="observable",
        aggfunc="first",
        fill_value=False,
    ).astype(bool)

    snap_labels = sorted(pivot.columns)
    if len(snap_labels) < 2:
        raise ValueError(
            f"Need at least 2 snapshot labels; got {snap_labels}"
        )
    last = snap_labels[-1]
    initials = snap_labels[:-1]

    obs_last = pivot[last]
    obs_initial = pivot[initials].any(axis=1)
    in_panel_mask = obs_last & obs_initial
    panel_cnpjs = set(in_panel_mask[in_panel_mask].index)

    def _cohort(cnpj: str) -> str:
        for s in snap_labels:
            if bool(pivot.loc[cnpj, s]):
                return s.split("_")[-1]
        return "unknown"

    cohort_map = {c: _cohort(c) for c in panel_cnpjs}

    panel = canon[canon["cnpj_clean"].isin(panel_cnpjs)].copy()
    panel["in_panel"] = True
    panel["cohort"] = panel["cnpj_clean"].map(cohort_map)
    panel["rota_tag"] = panel["rota"].apply(_rota_tag)

    for col in LONG_COLUMNS:
        if col not in panel.columns:
            panel[col] = pd.NA

    out = (
        panel[list(LONG_COLUMNS)]
        .sort_values(["cnpj_clean", "snapshot_date"])
        .reset_index(drop=True)
    )

    duration = time.time() - t_start

    if audit:
        run_log.record(
            step="renovabio.reconcile_and_filter",
            duration_s=duration,
            params={
                "geography": geography,
                "biofuel": biofuel,
                "option": option,
                "n_routes": len(routes),
            },
            input_rows=len(raw),
            output_rows=len(out),
            output_hash=hash_dataframe(out),
            notes=(
                f"{len(panel_cnpjs)} mills in panel "
                f"(observable at {last} ∧ observable at any of {initials})"
            ),
        )

    return out


def to_wide_with_deltas(
    panel_long: pd.DataFrame,
    outcomes: Iterable[OutcomeSpec],
    snapshot_label_order: Iterable[str] | None = None,
    entity_id_column: str = "cnpj_clean",
    audit: bool = True,
) -> pd.DataFrame:
    """Pivot long panel to wide and compute cohort-aware deltas.

    For each outcome:
      - {snapshot}_{outcome} columns are filled from the long observations.
      - {outcome}_first = first observable value (cohort-aware).
      - {outcome}_first_origin = which snapshot was used as initial reference.
      - delta_{outcome} = {outcome}_last − {outcome}_first.

    The output meets terrdiag's canonical wide PanelSchema.

    Parameters
    ----------
    panel_long : pd.DataFrame
        Output of `reconcile_and_filter`.
    outcomes : iterable of OutcomeSpec
        Outcomes for which to compute deltas. Each spec's snapshots dict
        determines which raw column from the long format maps to which
        snapshot label.
    snapshot_label_order : iterable of str | None
        Order of snapshots from earliest to latest. Default: sorted unique.
    entity_id_column : str
        Column to pivot on (default 'cnpj_clean').
    audit : bool
        Emit a run-log entry.

    Returns
    -------
    pd.DataFrame
        Wide panel meeting terrdiag.schema.PanelSchema.
    """
    t0 = time.time()
    outcomes = list(outcomes)

    if snapshot_label_order is None:
        snapshot_label_order = sorted(panel_long["snapshot_label"].unique())
    snapshot_label_order = list(snapshot_label_order)

    # Identifier columns (one row per entity)
    id_cols = (
        panel_long.drop_duplicates(entity_id_column)[
            [entity_id_column, "emissor", "cidade", "uf", "rota", "rota_tag", "cohort"]
        ]
        .reset_index(drop=True)
    )

    wide = id_cols.copy()

    # Pivot each measurable
    for raw_col in ["neea", "vol_pct"]:
        p = panel_long.pivot_table(
            index=entity_id_column,
            columns="snapshot_label",
            values=raw_col,
            aggfunc="first",
        )
        p.columns = [f"{raw_col}_{c}" for c in p.columns]
        wide = wide.merge(p.reset_index(), on=entity_id_column, how="left")

    # Pivot string fields for audit
    for raw_col in ["status", "cat_motivo", "processo"]:
        p = panel_long.pivot_table(
            index=entity_id_column,
            columns="snapshot_label",
            values=raw_col,
            aggfunc="first",
        )
        prefix = "motivo" if raw_col == "cat_motivo" else raw_col
        p.columns = [f"{prefix}_{c}" for c in p.columns]
        wide = wide.merge(p.reset_index(), on=entity_id_column, how="left")

    obs_p = panel_long.pivot_table(
        index=entity_id_column,
        columns="snapshot_label",
        values="observable",
        aggfunc="first",
        fill_value=False,
    )
    obs_p.columns = [f"observable_{c}" for c in obs_p.columns]
    wide = wide.merge(obs_p.reset_index(), on=entity_id_column, how="left")

    # Cohort-aware first/delta per outcome
    # NOTE: 'last' snapshot per the spec.snapshots ordering ('t2')
    initials_order = snapshot_label_order[:-1]
    last_label = snapshot_label_order[-1]

    for spec in outcomes:
        # We need to know which raw measurable corresponds to this outcome.
        # In RenovaBio: spec.name='NEEA' uses raw 'neea'; spec.name='vol_pct' uses 'vol_pct'.
        # The mapping is: spec.snapshots['t2'] gives us the target column name (e.g. 'neea_t2_2026').
        # We assume the pattern is '<raw_col>_<snapshot_label>'.
        sample_col = spec.snapshots.get("t2") or next(iter(spec.snapshots.values()))
        # Strip the snapshot label suffix to recover the raw column name
        last_suffix = f"_{last_label}"
        if sample_col.endswith(last_suffix):
            raw_col = sample_col[: -len(last_suffix)]
        else:
            raw_col = sample_col.rsplit("_", 1)[0]

        def _first(row: pd.Series, raw=raw_col, ini=initials_order):
            for s in ini:
                val = row.get(f"{raw}_{s}")
                obs = row.get(f"observable_{s}", False)
                if pd.notna(val) and bool(obs):
                    return val, s
            return np.nan, None

        firsts = wide.apply(_first, axis=1)
        wide[spec.first_column] = firsts.apply(lambda x: x[0])
        wide[spec.first_origin_column] = firsts.apply(lambda x: x[1])
        wide[spec.delta_column] = wide[f"{raw_col}_{last_label}"] - wide[spec.first_column]

    # Final column order
    base_cols = [entity_id_column, "emissor", "cidade", "uf", "rota", "rota_tag", "cohort"]

    outcome_cols: list[str] = []
    for spec in outcomes:
        outcome_cols.extend(list(spec.snapshots.values()))
        outcome_cols.extend([spec.first_column, spec.first_origin_column, spec.delta_column])

    audit_cols: list[str] = []
    for prefix in ["status", "motivo", "processo", "observable"]:
        for s in snapshot_label_order:
            col = f"{prefix}_{s}"
            if col in wide.columns:
                audit_cols.append(col)

    all_cols = base_cols + outcome_cols + audit_cols
    wide = wide[[c for c in all_cols if c in wide.columns]].sort_values(entity_id_column).reset_index(
        drop=True
    )

    duration = time.time() - t0

    if audit:
        run_log.record(
            step="renovabio.to_wide_with_deltas",
            duration_s=duration,
            params={
                "outcomes": [s.name for s in outcomes],
                "snapshot_label_order": snapshot_label_order,
                "entity_id_column": entity_id_column,
            },
            input_rows=len(panel_long),
            output_rows=len(wide),
            output_hash=hash_dataframe(wide),
            notes=f"{wide.shape[1]} columns, deltas computed for {len(outcomes)} outcomes",
        )

    return wide
