# case_studies/renovabio

RenovaBio panel construction. **Not part of the terrdiag library.** This
package transforms raw ANP certification spreadsheets into the canonical
wide panel format that terrdiag consumes.

## What lives here

| File | Role |
|---|---|
| `anp_parser.py` | Parser for ANP xlsx snapshots. Encodes three historical parsing-bug fixes. |
| `reconcile.py`  | `reconcile_and_filter` (Opção 2) + `to_wide_with_deltas` (pivot + cohort-aware deltas). |
| `presets.py`    | RenovaBio constants: snapshots, geography (Centro-Sul), routes (E1GC cana 1G), outcome specs (NEEA, vol%). |
| `make_panel.py` | End-to-end orchestration. Reads xlsx → produces canonical CSV. |

## Why this is not in `terrdiag/`

The terrdiag library is generic: it accepts panels that meet `PanelSchema`
and applies K-means typology + LISA + diagnostic cells. The parsing of
ANP xlsx, the Opção 2 reconciliation, and the Centro-Sul/E1GC filters are
specific to this case study and do not generalize. Keeping them outside
`src/terrdiag/` keeps the library clean and the case fully reproducible.

## Usage

### As a function

```python
from terrdiag.case_studies.renovabio import make_panel

wide_panel = make_panel.run(
    snapshots_dir="/path/to/anp/xlsx",
    gdf_pts_path="/path/to/gdf_pts_vol_corrected_cnpj.csv",
    output_path="painel_canonico.csv",
)
```

### As a CLI script

```bash
python -m terrdiag.case_studies.renovabio.make_panel \
  --snapshots-dir /path/to/anp \
  --gdf-pts /path/to/gdf_pts.csv \
  --output painel_canonico.csv
```

The script writes both the panel CSV and a `painel_canonico.metadata.json`
alongside, capturing input hashes, library version, and the run log.

## Subsequent analysis (using the panel from terrdiag)

```python
from terrdiag import io
from terrdiag.case_studies.renovabio.make_panel import RENOVABIO_SCHEMA
from terrdiag.case_studies.renovabio.presets import RENOVABIO_OUTCOMES

panel = io.load_panel(
    "painel_canonico.csv",
    schema=RENOVABIO_SCHEMA,
    outcomes=RENOVABIO_OUTCOMES,
)
# panel is now ready for terrdiag.layer1, terrdiag.layer2, etc.
```

## Bug-fix invariants (encoded inline)

The parser corrects three historical bugs from earlier iterations. Each is
marked with a `[BUG N FIX]` comment in `anp_parser.py`:

1. **Forward-fill scope** — `cert_idx` is regenerated per sheet and ffill
   is applied only by `cert_idx`, never by `(snapshot, status, cert_idx)`.
2. **UF extraction** — UF is extracted from `Endereço Emissor Primário`
   first (always present), with fallback to razão social, then resolved
   by `mode()` per CNPJ. Earlier versions used regex on the razão social
   only and silently dropped 60 mills in the 2026 snapshot.
3. **Embedded whitespace** — All string fields are normalized
   (`\r`, `\n`, `\t` → space, multi-space → single space) before any
   downstream operation. ANP spreadsheets contain linebreaks inside razão
   social cells as visual formatting; without normalization, CSV exports
   render with broken rows in spreadsheet viewers.

## Canonical panel — invariants

The output of `make_panel.run` with the default arguments must satisfy:

- 230 mills
- 674 observations (one row per mill × observable snapshot)
- UF distribution: SP 123 · GO 33 · MG 29 · PR 19 · MS 17 · MT 9
- 228 mills with X/Y from `gdf_pts`
- Cohorts: 214 mills (2022) · 16 mills (2023)
- NEEA complete in 3 snapshots: 213 mills
- vol(%) complete in 3 snapshots: 208 mills
- 207 mills with valid cert in 2026 · 23 mills in renewal transition
