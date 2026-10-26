# terrdiag

[![tests](https://github.com/alvarocoutinho/terrdiag/actions/workflows/tests.yml/badge.svg)](https://github.com/alvarocoutinho/terrdiag/actions/workflows/tests.yml)
[![Status: Beta](https://img.shields.io/badge/status-beta-orange.svg)](CHANGELOG.md)
[![Python: 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Two-layer territorial diagnostic framework for longitudinal public policy metrics.**

`terrdiag` combines an entity-level territorial typology (Layer 1: PCA + K-means) with point-level spatial autocorrelation (Layer 2: global and local Moran's I), and cross-tabulates both into diagnostic cells that flag tensions between territorial structure and outcome dynamics.

The library API is **policy-agnostic**: it consumes panels that meet a canonical schema and applies the analysis. Reference data-preparation pipelines for specific policy programmes ship as `terrdiag.case_studies.*` subpackages.

## Repository layout

```
terrdiag/
├── src/terrdiag/
│   ├── schema.py          ← PanelSchema + validate_panel
│   ├── io/                ← load_panel, write_panel, resolve_geo
│   ├── specs.py           ← FeatureSpec, OutcomeSpec, LISARecipe, ...
│   ├── presets.py         ← DEFAULT_LISA (generic defaults only)
│   ├── audit/             ← run log, hashing, reproducibility metadata
│   ├── layer1/            ← preprocess, impute, cluster, independence, inspect, plotting
│   ├── layer2/            ← weights, moran, lisa, sensitivity, inspect, plotting
│   ├── overlay/           ← diagnostic cells, inspect, plotting
│   ├── plotting/          ← placeholder (plots live in each layer)
│   └── case_studies/
│       └── renovabio/     ← Brazilian biofuel (CBIO) programme: ANP parsing → canonical panel
├── examples/              ← 01 generic smoke test · 02–05 RenovaBio pipeline, Layer 1, Layer 2
└── tests/
```

## Status

| Component | Status |
|---|---|
| Schema, validation, I/O | ✅ |
| Specs, presets, audit | ✅ |
| Layer 1 — entity typology (PCA + K-means, bootstrap ARI) | ✅ |
| Layer 2 — global and local Moran's I, weights, bandwidth sensitivity | ✅ |
| Overlay — diagnostic cells, association test | ✅ |
| `case_studies.renovabio` | ✅ |
| Top-level `plotting` module and API documentation | 🚧 |

The public API may change before 1.0. See [CHANGELOG.md](CHANGELOG.md).

## Installation

From GitHub (local terminal):

```bash
pip install "terrdiag[full] @ git+https://github.com/alvarocoutinho/terrdiag.git@v0.5.0b7"
```

In Google Colab:

```python
!pip install -q "terrdiag[full] @ git+https://github.com/alvarocoutinho/terrdiag.git@v0.5.0b7"
```

From a local copy of the source (for development):

```bash
pip install -e ".[full,dev]"
```

Extras: `terrdiag` (core only), `terrdiag[layer1]`, `terrdiag[layer2]`, `terrdiag[geo]`, `terrdiag[full]`, `terrdiag[dev]`.

The `terrdiag.case_studies.*` subpackages are included in every install.

## Usage — generic researcher

Bring your own panel that meets the canonical schema (one row per entity; columns `entity_id`, `cohort`; optional `X`/`Y`; plus longitudinal outcome columns).

```python
from terrdiag import io, schema, specs

my_schema = schema.PanelSchema(entity_id_column="entity_id")
my_outcome = specs.OutcomeSpec(
    name="evasao",
    snapshots={"t0": "evasao_2019", "t1": "evasao_2021", "t2": "evasao_2023"},
    delta_column="delta_evasao",
    first_column="evasao_first",
    first_origin_column="evasao_first_origin",
    units="%",
    direction="lower_better",
)
panel = io.load_panel("meu_painel.csv", schema=my_schema, outcomes=[my_outcome])
panel = io.resolve_geo(panel, points_source="meus_pontos.csv", schema=my_schema)
```

Then, for example, test spatial autocorrelation of the outcome (Layer 2):

```python
from terrdiag import layer2, overlay

W = layer2.gaussian_kernel_weights(panel["X"], panel["Y"], bandwidth_km=100)
mi = layer2.moran_i(panel["delta_evasao"], W, attribute_name="Δ evasão", bandwidth_km=100)
lisa = layer2.local_moran(panel["delta_evasao"], W, attribute_name="Δ evasão", bandwidth_km=100)
```

See `examples/01_generic_smoke.py` for a runnable synthetic example, and the module docstrings of `terrdiag.layer1`, `terrdiag.layer2` and `terrdiag.overlay` for the full API.

## Usage — RenovaBio case study

Two steps. Step 1 is RenovaBio-specific (lives in `terrdiag.case_studies.renovabio`); step 2 is generic terrdiag.

```python
# === Step 1: build the canonical panel from raw ANP xlsx ===
from terrdiag.case_studies.renovabio import make_panel
wide = make_panel.run(
    snapshots_dir="/path/to/anp",
    gdf_pts_path="/path/to/gdf_pts.csv",
    output_path="painel_canonico.csv",
)

# === Step 2: analysis with terrdiag (any subsequent session) ===
from terrdiag import io
from terrdiag.case_studies.renovabio.make_panel import RENOVABIO_SCHEMA
from terrdiag.case_studies.renovabio.presets import RENOVABIO_OUTCOMES

panel = io.load_panel(
    "painel_canonico.csv",
    schema=RENOVABIO_SCHEMA,
    outcomes=RENOVABIO_OUTCOMES,
)
```

See `src/terrdiag/case_studies/renovabio/README.md` for the RenovaBio data-preparation details and bug-fix invariants. Raw ANP spreadsheets are not distributed with this repository.

## Reproducibility

Every high-level call emits a run-log entry and contributes to `run_metadata.json`:

```python
from terrdiag import audit
# ... calls to io / layer1 / layer2 ...
audit.write_metadata("run_metadata.json", inputs={...})
```

The metadata captures library version, Python version, platform, SHA-256 hashes of input files, parameters used, and timing per step.

## Philosophy

1. **Pure functions > OOP** for analysis primitives; immutable dataclasses for configuration.
2. **The library API is policy-agnostic.** Case-specific code lives in `case_studies/`, clearly namespaced.
3. **Reproducibility orthogonal to user effort.** No manual log/hash calls required.
4. **`geopandas` optional.** Plain X/Y in DataFrames is the default path.

## Citation

If you use terrdiag, please cite the archived version you used. GitHub's **"Cite this repository"** button (generated from [`CITATION.cff`](CITATION.cff)) gives APA and BibTeX entries, and every release is archived on Zenodo with its own DOI.

Coutinho, Á. L. A. (2026). *terrdiag: Two-layer territorial diagnostic framework* (Version 0.5.0b7) [Computer software]. https://github.com/alvarocoutinho/terrdiag

## Acknowledgements

This study was financed in part by the Coordenação de Aperfeiçoamento de Pessoal de Nível Superior – Brasil (CAPES) – Finance Code 001, and by grant #2025/01530-0, São Paulo Research Foundation (FAPESP). Developed at the Graduate Program in Applied Ecology (PPGI-EA), ESALQ/CENA, Universidade de São Paulo.

## License

[MIT](LICENSE).
