"""
terrdiag.case_studies.renovabio.anp_parser — ANP RenovaBio snapshot parser.

This module is the frozen parser for ANP certification spreadsheets.
Three historical parsing bugs are corrected here, with inline comments:

    [BUG 1 FIX] forward-fill scope: by cert_idx APENAS, regenerated per sheet.
    [BUG 2 FIX] UF extraction: from Endereço Emissor Primário with fallback
                to razão social, then mode-per-CNPJ.
    [BUG 3 FIX] embedded whitespace: clean \\r, \\n, \\t from all string
                fields before downstream operations.

Output: a long-format raw DataFrame with one row per
(cnpj, snapshot, status, biocomb) observation, ready to be filtered and
reconciled by `reconcile.py`.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from terrdiag.audit.hashing import hash_dataframe
from terrdiag.audit.logging import run_log


# ─── Snapshot descriptor ──────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class SnapshotSource:
    """One ANP snapshot input.

    Attributes
    ----------
    path : str
        Filesystem path (file name; combined with base_path at load time).
    date : str
        ISO 8601 snapshot date ('YYYY-MM-DD').
    label : str
        Canonical short label used downstream ('t0_2022', etc.).
    sheets : tuple of (sheet_name, status_label)
        Sheets to extract; default covers the three ANP categories.
    """

    path: str
    date: str
    label: str
    sheets: tuple[tuple[str, str], ...] = (
        ("Válidos", "valido"),
        ("Cancelados ou Suspensos", "cancelado"),
        ("Anulados", "anulado"),
    )

    def __post_init__(self) -> None:
        if len(self.date) != 10 or self.date[4] != "-" or self.date[7] != "-":
            raise ValueError(f"date must be ISO 8601 (YYYY-MM-DD), got {self.date!r}")
        if not self.label or not self.label.startswith("t"):
            raise ValueError(f"label must start with 't', got {self.label!r}")


# ─── Constants ────────────────────────────────────────────────────────────


_UF_PATTERN = re.compile(r"(?<![A-Z])([A-Z]{2})(?![A-Z])")
_CITY_PATTERN = re.compile(r"-\s*([^-/]+?)\s*[-/]\s*[A-Z]{2}")
_CNPJ_DIGITS = re.compile(r"\D")

_FFILL_COLS = (
    "Emissor primário",
    "CNPJ",
    "Processo de Certificação",
    "Rota",
    "Motivo",
    "Data de Aprovação pela ANP",
    "Endereço Emissor Primário",
)


# ─── Helpers ──────────────────────────────────────────────────────────────


def _clean_whitespace(s: object) -> object:
    """[BUG 3 FIX] Replace \\r \\n \\t with space; collapse repeats."""
    if pd.isna(s):
        return s
    s2 = str(s).replace("\r", " ").replace("\n", " ").replace("\t", " ")
    return re.sub(r"\s+", " ", s2).strip()


def _normalize_cnpj(s: object) -> str | None:
    if pd.isna(s):
        return None
    digits = _CNPJ_DIGITS.sub("", str(s))
    return digits.zfill(14) if digits else None


def _extract_uf(emissor: object, endereco: object, valid_ufs: frozenset[str]) -> str | None:
    """[BUG 2 FIX] UF extraction from address first, then razão social."""
    for src in (endereco, emissor):
        if pd.isna(src):
            continue
        for m in _UF_PATTERN.findall(str(src)):
            if m in valid_ufs:
                return m
    return None


def _extract_city(emissor: object, endereco: object) -> str | None:
    for src in (emissor, endereco):
        if pd.isna(src):
            continue
        m = _CITY_PATTERN.search(str(src))
        if m:
            city = m.group(1).strip()
            if 1 < len(city) < 60 and not any(
                t in city for t in ("CEP", "Rod", "Km", "S/n", "Fazenda")
            ):
                return city
    return None


def _categorize_motivo(m: object) -> str:
    if pd.isna(m):
        return "sem_motivo"
    s = str(m).lower()
    if "renova" in s:
        return "renovacao"
    if "expir" in s or "vencid" in s:
        return "expirado"
    if "pedido" in s or "solicit" in s:
        return "pedido_emissor"
    if "recuper" in s:
        return "rec_judicial"
    return "outros"


def _parse_sheet(
    path: str | Path,
    sheet: str,
    snapshot_date: str,
    snapshot_label: str,
    status: str,
) -> pd.DataFrame:
    """[BUG 1 FIX] cert_idx regenerated per sheet, ffill only by cert_idx."""
    df = pd.read_excel(path, sheet_name=sheet, header=1)
    df = df.dropna(how="all").reset_index(drop=True)
    df.columns = ["seq"] + list(df.columns[1:])

    if "Razão Social - Cidade - UF" in df.columns and "Emissor primário" not in df.columns:
        df = df.rename(columns={"Razão Social - Cidade - UF": "Emissor primário"})

    df["cert_idx"] = df["seq"].notna().cumsum()
    for col in _FFILL_COLS:
        if col in df.columns:
            df[col] = df.groupby("cert_idx")[col].ffill()

    df["snapshot_date"] = snapshot_date
    df["snapshot_label"] = snapshot_label
    df["status"] = status
    return df


# ─── Public API ───────────────────────────────────────────────────────────


def load_anp_raw(
    snapshots: Iterable[SnapshotSource],
    valid_ufs: frozenset[str] | Iterable[str],
    base_path: str | Path | None = None,
    audit: bool = True,
) -> pd.DataFrame:
    """Load and concatenate ANP snapshot spreadsheets.

    Parameters
    ----------
    snapshots : iterable of SnapshotSource
    valid_ufs : iterable of str
        UFs considered valid during the UF-extraction mode resolution
        (BUG 2 FIX); typically the 27 Brazilian UFs from
        `terrdiag.case_studies.renovabio.presets.VALID_BRAZILIAN_UFS`.
    base_path : str | Path | None
        If provided, snapshot.path is resolved as base_path / snapshot.path.
    audit : bool
        Emit a run-log entry. Default True.

    Returns
    -------
    pd.DataFrame
        Long-format raw frame with one row per (cnpj, snapshot, status,
        biocomb) observation. Columns include cnpj_clean, snapshot_label,
        status, emissor, cidade, uf, rota, processo, neea, vol_pct,
        cat_motivo.
    """
    t0 = time.time()
    snapshots = list(snapshots)
    if not snapshots:
        raise ValueError("snapshots is empty")
    valid_ufs = (
        valid_ufs if isinstance(valid_ufs, frozenset) else frozenset(valid_ufs)
    )

    frames: list[pd.DataFrame] = []
    for snap in snapshots:
        path = Path(base_path) / snap.path if base_path else Path(snap.path)
        if not path.exists():
            # Build a helpful error: list .xlsx files actually present in the directory
            search_dir = Path(base_path) if base_path else path.parent
            hint = ""
            if search_dir.exists() and search_dir.is_dir():
                xlsx_files = sorted(p.name for p in search_dir.glob("*.xlsx"))
                if xlsx_files:
                    hint = (
                        f"\n\nFiles ending in .xlsx found in {search_dir}:\n  "
                        + "\n  ".join(xlsx_files)
                        + "\n\nIf the file names differ from the defaults, build your own "
                        "SnapshotSource tuple — see RENOVABIO_SNAPSHOTS in "
                        "terrdiag.case_studies.renovabio.presets for the template."
                    )
                else:
                    hint = f"\n\nDirectory {search_dir} exists but contains no .xlsx files."
            else:
                hint = (
                    f"\n\nDirectory not found: {search_dir}\n"
                    "Check that the Drive is mounted and the path is correct."
                )
            raise FileNotFoundError(f"Snapshot file not found: {path}{hint}")
        for sheet_name, status_label in snap.sheets:
            frames.append(_parse_sheet(path, sheet_name, snap.date, snap.label, status_label))

    df = pd.concat(frames, ignore_index=True)
    df = df[df["Biocombustível"].notna()].copy()

    df = df.rename(
        columns={
            "Nota de Eficiência Energético-Ambiental (gCO2eq/MJ)": "neea",
            "Volume elegível (%)": "vol_pct",
            "Rota": "rota",
            "Data de Aprovação pela ANP": "data_aprov_str",
            "Endereço Emissor Primário": "endereco",
            "Emissor primário": "emissor",
            "Processo de Certificação": "processo",
            "Biocombustível": "biocombustivel",
        }
    )

    df["data_aprov"] = pd.to_datetime(df["data_aprov_str"], errors="coerce")
    df["neea"] = pd.to_numeric(df["neea"], errors="coerce")
    df["vol_pct"] = pd.to_numeric(df["vol_pct"], errors="coerce")
    df["cnpj_clean"] = df["CNPJ"].apply(_normalize_cnpj)
    df["biocombustivel"] = df["biocombustivel"].astype(str).str.strip()

    # [BUG 3 FIX] Clean whitespace in ALL object columns before any extraction
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].apply(_clean_whitespace)

    # [BUG 2 FIX] UF: endereço → emissor → mode-per-CNPJ
    df["uf_raw"] = df.apply(
        lambda r: _extract_uf(r.get("emissor"), r.get("endereco"), valid_ufs), axis=1
    )
    uf_by_cnpj = (
        df[df["uf_raw"].notna()]
        .groupby("cnpj_clean")["uf_raw"]
        .agg(lambda x: x.mode().iloc[0])
        .to_dict()
    )
    df["uf"] = df["cnpj_clean"].map(uf_by_cnpj)

    df["cidade_raw"] = df.apply(
        lambda r: _extract_city(r.get("emissor"), r.get("endereco")), axis=1
    )
    cidade_by_cnpj = (
        df[df["cidade_raw"].notna()]
        .groupby("cnpj_clean")["cidade_raw"]
        .agg(lambda x: x.mode().iloc[0])
        .to_dict()
    )
    df["cidade"] = df["cnpj_clean"].map(cidade_by_cnpj)

    df["cat_motivo"] = df["Motivo"].apply(_categorize_motivo)

    df = df.drop(columns=["uf_raw", "cidade_raw", "cert_idx", "seq"], errors="ignore")

    duration = time.time() - t0

    if audit:
        run_log.record(
            step="renovabio.load_anp_raw",
            duration_s=duration,
            params={
                "n_snapshots": len(snapshots),
                "snapshot_labels": [s.label for s in snapshots],
                "base_path": str(base_path) if base_path else None,
            },
            output_rows=len(df),
            output_hash=hash_dataframe(df.head(5000)),
            notes=f"{df['cnpj_clean'].nunique()} unique CNPJs across all snapshots",
        )

    return df
