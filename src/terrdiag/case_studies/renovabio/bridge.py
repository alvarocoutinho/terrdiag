"""
terrdiag.case_studies.renovabio.bridge — Mill → IBGE municipality bridge.

Two responsibilities:
    1. Resolve cd_ibge for every mill in the canonical panel.
    2. Apply manual overrides for mills whose municipality extraction
       from ANP razão social/endereço failed (~14 known cases).

The bridge supports:
    - A user-supplied crosswalk CSV (geocode + muni_key)
    - A bundled fallback overrides table at:
        terrdiag/case_studies/renovabio/data/bridge_overrides.csv
"""

from __future__ import annotations

import re
import time
import unicodedata
from importlib import resources
from pathlib import Path

import pandas as pd

from terrdiag.audit.hashing import hash_dataframe
from terrdiag.audit.logging import run_log


# Package-bundled override CSV — resolves the 14 mills with ambiguous cities
DEFAULT_OVERRIDES_RESOURCE = ("terrdiag.case_studies.renovabio.data", "bridge_overrides.csv")


def _normalize_key(cidade: object, uf: object) -> str | None:
    """Normalize (cidade, uf) → 'CIDADE|UF' with accents and apostrophes stripped."""
    if pd.isna(cidade) or pd.isna(uf):
        return None
    s = unicodedata.normalize("NFD", str(cidade)).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[\s']+", " ", s).strip().upper()
    return f"{s}|{uf}"


def _normalize_crosswalk_key(s: object) -> str:
    """Normalize crosswalk muni_key the same way (handles apostrophe variants)."""
    s2 = re.sub(r"[\s']+", " ", str(s)).strip().upper()
    return s2


def load_default_overrides() -> pd.DataFrame:
    """Load the bundled bridge_overrides.csv (14 hand-resolved mills)."""
    pkg, fname = DEFAULT_OVERRIDES_RESOURCE
    with resources.files(pkg).joinpath(fname).open("r", encoding="utf-8-sig") as f:
        df = pd.read_csv(f, dtype=str)
    return df


def attach_cd_ibge(
    panel: pd.DataFrame,
    crosswalk_path: str | Path,
    overrides_path: str | Path | None = "default",
    audit: bool = True,
) -> pd.DataFrame:
    """Attach cd_ibge to each mill in the panel.

    Strategy:
        1. Normalize the panel's (cidade, uf) into a muni_key.
        2. Look it up in the crosswalk (which has geocode per muni_key).
        3. For mills where the lookup fails OR the panel's CNPJ is in the
           overrides table, replace cd_ibge with the override value.

    Parameters
    ----------
    panel : pd.DataFrame
        Canonical panel from terrdiag.io.load_panel.
    crosswalk_path : str | Path
        CSV with columns 'geocode', 'muni_key' (and optionally 'municipio', 'uf').
    overrides_path : str | Path | 'default' | None
        - 'default' (default): use the bundled overrides CSV.
        - Path: use the user's CSV (must have 'cnpj_clean', 'cd_ibge' columns).
        - None: do not apply any overrides.
    audit : bool
        Emit a run-log entry.

    Returns
    -------
    pd.DataFrame
        Panel with `cd_ibge` (str, 7 digits) column added.
    """
    t0 = time.time()
    out = panel.copy()

    # Step 1: build crosswalk lookup
    cw = pd.read_csv(crosswalk_path, dtype=str)
    if "muni_key" not in cw.columns or "geocode" not in cw.columns:
        raise ValueError(
            f"Crosswalk must have 'muni_key' and 'geocode' columns; "
            f"found {sorted(cw.columns)}"
        )
    cw["muni_key_norm"] = cw["muni_key"].apply(_normalize_crosswalk_key)
    cw_map = dict(zip(cw["muni_key_norm"], cw["geocode"]))

    # Step 2: panel muni_key → cd_ibge
    out["muni_key"] = out.apply(lambda r: _normalize_key(r.get("cidade"), r.get("uf")), axis=1)
    out["cd_ibge"] = out["muni_key"].map(cw_map)

    # Step 3: apply overrides
    n_overrides_applied = 0
    if overrides_path == "default":
        ov = load_default_overrides()
    elif overrides_path is None:
        ov = pd.DataFrame(columns=["cnpj_clean", "cd_ibge"])
    else:
        ov = pd.read_csv(overrides_path, dtype=str)

    if "cnpj_clean" not in ov.columns or "cd_ibge" not in ov.columns:
        if len(ov) > 0:
            raise ValueError(
                "Overrides CSV must have 'cnpj_clean' and 'cd_ibge' columns; "
                f"found {sorted(ov.columns)}"
            )

    ov_map = dict(zip(ov["cnpj_clean"].astype(str), ov["cd_ibge"].astype(str)))
    mask = out["cnpj_clean"].astype(str).isin(ov_map)
    n_overrides_applied = int(mask.sum())
    if n_overrides_applied > 0:
        out.loc[mask, "cd_ibge"] = out.loc[mask, "cnpj_clean"].astype(str).map(ov_map)

    # Drop intermediate
    out = out.drop(columns=["muni_key"], errors="ignore")

    n_unresolved = int(out["cd_ibge"].isna().sum())
    n_unique_munis = int(out["cd_ibge"].nunique())

    duration = time.time() - t0
    if audit:
        run_log.record(
            step="renovabio.attach_cd_ibge",
            duration_s=duration,
            params={
                "crosswalk_path": str(crosswalk_path),
                "overrides_path": str(overrides_path),
                "n_overrides_applied": n_overrides_applied,
            },
            input_rows=len(panel),
            output_rows=len(out),
            output_hash=hash_dataframe(out),
            notes=(
                f"{len(out) - n_unresolved}/{len(out)} rows with cd_ibge; "
                f"{n_unique_munis} unique municipalities; "
                f"{n_overrides_applied} overrides applied; "
                f"{n_unresolved} unresolved"
            ),
        )

    if n_unresolved > 0:
        import warnings as _w

        unresolved_cnpjs = out[out["cd_ibge"].isna()]["cnpj_clean"].unique().tolist()
        _w.warn(
            f"attach_cd_ibge: {n_unresolved} rows unresolved. "
            f"CNPJs: {unresolved_cnpjs[:10]}",
            stacklevel=2,
        )

    return out
