# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and version numbers
follow [PEP 440](https://peps.python.org/pep-0440/).

## [0.5.0b7] - 2026-10-26

First public release, archived on Zenodo.

### Included

- `schema` and `io`: canonical panel schema, validation, `load_panel`, `write_panel`, `resolve_geo`.
- `specs` and `presets`: `FeatureSpec`, `OutcomeSpec`, `LISARecipe` and generic defaults.
- `layer1`: preprocessing, imputation (median by group), PCA + K-means typology with bootstrap ARI, Kruskal–Wallis independence test, inspection tables and plots.
- `layer2`: Gaussian-kernel, k-nearest-neighbour and inverse-distance weights; global and local Moran's I; bandwidth sensitivity sweep; inspection tables and plots.
- `overlay`: Layer 1 × Layer 2 diagnostic cells, chi-squared association test, heatmap, Sankey and map plots.
- `audit`: run log, SHA-256 hashing of inputs and `run_metadata.json`.
- `case_studies.renovabio`: reference pipeline from ANP certification spreadsheets to the canonical panel.

### Known limitations

- The top-level `terrdiag.plotting` module is a placeholder; plots live in each layer.
- RenovaBio integration tests are skipped unless ANP fixtures are provided (`TERRDIAG_FIXTURES`).

[0.5.0b7]: https://github.com/alvarocoutinho/terrdiag/releases/tag/v0.5.0b7
